import abc
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd


from catsample import sample_categorical

def get_graph(config, device):
    if config.graph.type == "uniform":
        return Uniform(config.tokens)
    elif config.graph.type == "absorb":
        return Absorbing(config.tokens)
    elif config.graph.type == "mixed":
        return Mixed(config.tokens,config.lam,config.beta)
    else:
        raise ValueError(f"Graph {config.graph.type} not valid")


def unsqueeze_as(x, y, back=True):
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)


class Graph(abc.ABC):

    @property
    def dim(self):
        pass

    @property
    def absorb(self):
        """
        Whether input {dim - 1} is an absorbing state (used for denoising to always remove the mask).
        """
        pass


    @abc.abstractmethod
    def rate(self, i):
        """
        Computes the i-th column of the rate matrix Q, where i is [B_1, ..., B_n].

        This is intended to compute the "forward" rate of p(X_t | X_0 = i).
        """
        pass


    @abc.abstractmethod
    def transp_rate(self, i):
        """
        Computes the i-th row of the rate matrix Q.

        Can be used to compute the reverse rate.
        """
        pass


    @abc.abstractmethod
    def transition(self, i, sigma):
        """
        Computes the i-th column of the transition matrix e^{sigma Q}.
        """
        pass


    def sample_transition(self, i, sigma):
        """
        Samples the transition vector.
        """
        transition_vector = self.transition(i, sigma)
        return sample_categorical(transition_vector, method="hard")
    

    def reverse_rate(self, i, score,sigma):
        """
        Constructs the reverse rate. Which is score * transp_rate
        """
        normalized_rate = self.transp_rate(i) * score

        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate

    def sample_rate(self, i, rate,generator=None):
        return sample_categorical(F.one_hot(i, num_classes=self.dim).to(rate) + rate,generator=generator)

    
    @abc.abstractmethod
    def staggered_score(self, score, dsigma):
        """
        Computes p_{sigma - dsigma}(z) / p_{sigma}(x), which is approximated with
        e^{-{dsigma} E} score
        """
        pass
    

    @abc.abstractmethod
    def sample_limit(self, *batch_dims):
        """
        Sample the limiting distribution. Returns the probability vector as well.
        """
        pass


    @abc.abstractmethod
    def score_entropy(self, score, sigma, x, x0):
        """
        Computes the score entropy function (with requisite constant normalization)
        """
        pass


class Uniform(Graph):
    """
    Everything goes to everything else. Normalized down by dimension to avoid blowup.
    """
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim
    
    @property
    def absorb(self):
        return False

    def rate(self, i):
        edge = torch.ones(*i.shape, self.dim, device=i.device) / self.dim
        edge = edge.scatter(-1, i[..., None], - (self.dim - 1) / self.dim)
        return edge

    def transp_rate(self, i):
        return self.rate(i)

    def transition(self, i, sigma):
        trans = torch.ones(*i.shape, self.dim, device=i.device) * (1 - (-sigma[..., None]).exp()) / self.dim
        trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans))
        trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        return trans
    
    def transp_transition(self, i, sigma):
        return self.transition(i, sigma)

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, torch.randint_like(i, self.dim), i)
        return i_pert

    def staggered_score(self, score, dsigma):
        dim = score.shape[-1]
        epow = (-dsigma).exp()[..., None]
        return ((epow - 1) / (dim * epow)) * score.sum(dim=-1, keepdim=True) + score / epow

    def sample_limit(self, *batch_dims):
        return torch.randint(0, self.dim, batch_dims)

    def score_entropy(self, score, sigma, x, x0):
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        ratio = 1 - self.dim / (esigm1 + self.dim)

        # negative term
        neg_term = score.mean(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1) / self.dim
        # no move means scaling by the uniform ratio. move means alter only one ratio away from 1
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(score, -1, x0[..., None]).squeeze(-1) / esigm1 + neg_term
        )

        # constant factor
        const = torch.where(
            x == x0,
            (self.dim - 1) / self.dim * ratio * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim 
        )

        #positive term
        sexp = score.exp()
        pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / self.dim
        return pos_term - neg_term + const


class Absorbing(Graph):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim + 1
    
    @property
    def absorb(self):
        return True

    def rate(self, i):
        # edge = - F.one_hot(i, num_classes=self.dim)
        # edge.scatter_add_(-1, i[..., None], torch.ones_like(edge[..., :1]))
        return F.one_hot((self.dim - 1) * torch.ones_like(i), num_classes=self.dim) - F.one_hot(i, num_classes=self.dim)        

    def transp_rate(self, i):
        edge = -F.one_hot(i, num_classes=self.dim)
        edge[i == self.dim - 1] += 1
        return edge

    def transition(self, i, sigma):
        pass
    
    def transp_transition(self, i, sigma):
        sigma = unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)
        edge += torch.where(
            i == self.dim - 1,
            1 - (-sigma).squeeze(-1).exp(),
            0
        )[..., None]
        return edge

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, self.dim - 1, i)
        return i_pert
    
    def staggered_score(self, score, dsigma):
        score = score.clone() # yeah yeah whatever we should probably do this
        extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., -1] += extra_const
        return score

    def sample_limit(self, *batch_dims):
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)

    def score_entropy(self, score, sigma, x, x0):
        rel_ind = x == self.dim - 1
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )

        ratio = 1 / esigm1.expand_as(x)[rel_ind]
        other_ind = x0[rel_ind]

        # negative_term
        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)

        #positive term
        pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)

        # constant term
        const = ratio * (ratio.log() - 1)

        entropy = torch.zeros(*x.shape, device=x.device)
        entropy[rel_ind] += pos_term - neg_term + const
        return entropy
    
class Mixed(Graph):
    def __init__(self, dim,beta,lam):
        super().__init__()
        self._dim = dim
        self.beta=beta 
        self.lam=lam

    @property
    def dim(self):
        return self._dim + 1
    
    @property
    def absorb(self):
        return False
    
    @property
    def mixed(self):
        return True
    
    def transition(self, i, sigma):
        """
        Computes the i-th column of the transition matrix e^{sigma Q}.
        """
        print('Need to implement this')
        pass

    def sample_transition(self, i, sigma):
        mask_chance = 1 - (-self.beta*sigma).exp()
        mask_indices = torch.rand(*i.shape, device=i.device) < mask_chance
        i_pert = torch.where(mask_indices, self.dim - 1, i)
        flip_chance=1 - (-self.lam*sigma).exp()
        flip_indices=torch.rand(*i.shape, device=i.device) < flip_chance
        i_pert = torch.where((flip_indices)& (~mask_indices), torch.randint_like(i, self._dim), i)
        return i_pert
    
    def sample_limit(self, *batch_dims):
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)
    
    def rate(self,i):
        edge = self.lam * torch.ones(*i.shape, self.dim, device=i.device)
        edge[..., -1] = self.beta
        edge = edge.scatter(-1, i[..., None], -self.lam * (self.dim - 2) - self.beta)

        mask = (i == self.dim-1)[..., None]  
        edge = edge.masked_fill(mask, 0)
        
        return edge

    def transp_rate(self, i, f_t):
        # i: (B, L), f_t: (B, 1)
        B, L = i.shape
        edge = self.lam * torch.ones(B, L, self.dim, device=i.device)
        edge[..., -1] = 0
        edge = edge.scatter(-1, i[..., None], -self.lam * (self.dim - 2) - self.beta)

        # Mask where i == dim - 1
        mask = (i == self.dim - 1)  # shape (B, L)

        # Compute per-batch fill values
        fill = (f_t * self.beta).view(B, 1, 1)  # shape (B, 1, 1), broadcastable to (B, L, dim)

        # Use broadcasting to fill edge where mask is True
        edge = torch.where(mask[..., None], fill.expand(-1, L, self.dim), edge)

        return edge

    
    def reverse_rate(self, i, score,sigma):
        """
        Constructs the reverse rate. Which is score * transp_rate
        """
        N = self.dim - 1  # real tokens in [0..N-1], 'mask' is ID=N
        lam = self.lam
        beta=self.beta
        a_t = torch.exp(-N * lam * sigma)    # (1-t)^(N*lam)
        b_t = torch.exp(-beta * sigma)       # (1-t)^beta

        # shape (B,) or scalar if B=1
        denom_f = (1.0 - b_t)*N + 1e-12
        f_t = (b_t*(1.0 - a_t))/denom_f

        normalized_rate = self.transp_rate(i,f_t) * score

        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate
    
    def staggered_score(self, score, dsigma):
        """
        Computes p_{sigma - dsigma}(z) / p_{sigma}(x), which is approximated with
        e^{-{dsigma} E} score
        """
        print('Need to implement this')
        pass


    def score_entropy(self,
        Mbar,
        bar_sigma,
        x_t,        # (B, L) the noised sample in [0..N]  (N is mask)
        x_data,     # (B, L) the original data in [0..N-1]
        ):
        r"""
        Matching logic:
        1) a(t) = (1-t)^(N*lam), b(t) = (1-t)^beta, sigma(t) = 1/(1-t)
        2) f(t) = [b(t)*(1 - a(t))]/[(1 - b(t))*N], lamBar=log(1 + N*a/(1 - a))
        3) For each token i in each batch:
            - if x_t[i] == N => factor = f(t)*beta
                toexp = lamBar if x_data[i]==y else 0
            else => factor = lam
                toexp = lamBar if x_data[i]==y
                        -lamBar if x_data[i]==x_t[i]
                        else 0
            - skip y==x_t[i]
            - sum ell(Mbar[y], toexp) where ell(x,y)= e^x - e^y + e^y*(y - x)
        4) Multiply total sum by sigma(t), and average over B*L.
        """

        device = x_data.device
        B, L = x_data.shape
        N = self.dim - 1  # real tokens in [0..N-1], 'mask' is ID=N
        lam = self.lam
        beta=self.beta
        # 1) a,b,sigma,f,lamBar
        a_t = torch.exp(-N * lam * bar_sigma)    # (1-t)^(N*lam)
        b_t = torch.exp(-beta * bar_sigma)       # (1-t)^beta

        # shape (B,) or scalar if B=1
        denom_f = (1.0 - b_t)*N + 1e-12
        f_t = (b_t*(1.0 - a_t))/denom_f

        denom_lb = (1.0 - a_t) + 1e-12
        lamBar_t = torch.log(1.0 + (N*a_t)/denom_lb)

        # 3) We sum over y in [0..N-1], skipping y=x_t if x_t < N
        # build a grid of candidate y
        candidates = torch.arange(N, device=device).view(1,1,N)  # shape (1,1,N)

        # skip_mask= True if y==x_t => skip.
        # If x_t==N => none of these are skipped, because y in [0..N-1], so y==N never matches
        skip_mask = (candidates == x_t.unsqueeze(-1))  # shape (B,L,N)

        # x_t_is_mask => shape (B,L), true if x_t==N
        x_t_is_mask = (x_t == N)  # (B,L)
        # define factor= f(t)*beta if mask else lam
        # shape (B,L)
        factor_2d = torch.where(x_t_is_mask, f_t*beta, lam*torch.ones_like(x_t, dtype=torch.float32))
        # expand to (B,L,N)
        factor_3d = factor_2d.unsqueeze(-1).expand(B, L, N)

        # define toexp piecewise
        #   if x_t==N => lamBar if x_data==y else 0
        #   else => lamBar if x_data==y
        #           -lamBar if x_data==x_t
        #           else 0
        x_data_eq_y  = (x_data.unsqueeze(-1) == candidates)   # (B,L,N)
        x_data_eq_xt = (x_data == x_t).unsqueeze(-1)          # (B,L,1)

        # We'll need lamBar broadcast to (B,L,N).
        # lamBar_t is shape (B,). We'll expand to (B,1,1) => then expand to (B,L,N).
        lamBar_expand = lamBar_t.view(-1,1,1).expand(B,L,N)

        zero_3d = torch.zeros((B,L,N), dtype=torch.float32, device=device)

        # if x_t==N => toexp= lamBar where x_data_eq_y else 0
        # shape (B,L,N)
        toexp_mask_case = torch.where(x_data_eq_y, lamBar_expand, zero_3d)

        # if x_t!=N => toexp= lamBar where x_data_eq_y, -lamBar where x_data_eq_xt, else 0
        # so first define a version that sets lamBar if x_data_eq_y else 0
        toexp_temp = torch.where(x_data_eq_y, lamBar_expand, zero_3d)
        # then we override where x_data_eq_xt is True => -lamBar
        toexp_nonmask_case = torch.where(x_data_eq_xt, -lamBar_expand, toexp_temp)

        # Now pick between mask or non-mask
        x_t_is_mask_3d = x_t_is_mask.unsqueeze(-1)  # shape (B,L,1)
        toexp = torch.where(x_t_is_mask_3d, toexp_mask_case, toexp_nonmask_case)

        # 4) ell(Mbar[y], toexp):
        #    ell(x,y)= e^x - e^y + e^y*(y - x)
        Mbar_y   = Mbar[...,:N]         # shape (B,L,N), ignoring the last index (mask)
        exp_Mbar = torch.exp(Mbar_y)    # shape (B,L,N)
        exp_toexp= torch.exp(toexp)     # shape (B,L,N)
        ell_val  = exp_Mbar - exp_toexp + exp_toexp*(toexp - Mbar_y)

        # skip transitions where y=x_t
        skip_flt = (~skip_mask).float()

        # local sum => factor_3d * ell_val * skip_flt
        local_sum = factor_3d * ell_val * skip_flt

        # multiply sum_over_y by sigma(t) => need shape (B,L) for sigma
        # sigma_t is shape (B,). expand to (B,L)
        sum_over_y = local_sum.sum(dim=-1) 

        return sum_over_y