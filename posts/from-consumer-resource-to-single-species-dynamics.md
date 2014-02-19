
## From consumer-resource models to single-species dynamics

We are interested in building simple consumer-resource models and exploring how
single-species models emerge as we simplify the dynamics of the resource. We
assume a simple Lotka-Volterra type equation, with a carrying capacity for the
resource:

$$ \begin{aligned}
    \frac{dR}{dt} &= rR \left(1-\frac{R}{K}\right) - cRN \\
    \frac{dN}{dt} &= N(ecR - d)
\end{aligned}$$
The fixed points are: the trivial one $(0, 0)$, only resource $(K, 0)$, and the
coexistence one, given by $R^* = \frac{d}{ec}$, $N^* =
\frac{r}{c}\left(1-\frac{R^*}{K}\right)$. The trivial fixed point is unstable
(the resource is always able to grow), and the fixed point without consumers is
unstable if $ecK-d > 0$, that is, if the consumer is able to grow when the
resource is at its carrying capacity. The stability of the coexistence fixed
point can be found calculating the Jacobian at that point:

$$ J = \begin{bmatrix}
    -\frac{r}{K} & -cR^* \\
    ecN^*        & 0
  \end{bmatrix}  =
  \begin{bmatrix}
    -\frac{r}{K}                   & -\frac{d}{e} \\
    er\left(1-\frac{R^*}{K}\right) & 0
  \end{bmatrix}
$$

We can see directly that the trace $T$ of the Jacobian is negative while its
determinant $\Delta$ is positive. The eigenvalues $\lambda$ are determined by $
\lambda^2 - T \lambda + \Delta = 0 $, so $ \lambda = \frac{1}{2}\left[T \pm
\sqrt{T^2 - 4\Delta} \right] $. It is clear that the real part of the
eigenvalues are always negative, and the eigenvalues will be complex when the
term inside the square root becomes negative. Thus, we conclude that the
coexistence fixed point is always stable.

Now we rescale the variables and time by letting $t' = dt$, $R'=\frac{R}{R^*}$
and $N'=\frac{N}{N^*}$. Rewriting the equations (and dropping the primes), we
have:

$$ \begin{aligned}
    \frac{dR}{dt} &= \frac{r}{d} \left[ R (1-\beta R) - (1-\beta) RN \right]\\
    \frac{dN}{dt} &= N (R - 1)
\end{aligned} $$

where $\beta = \frac{R^*}{K} = \frac{d}{ecK}$.

Let's assume that the resource growth rate is much faster than the consumer's
death rate, that is, $r \gg d$. Another way of writing this is $\frac{d}{r} =
\epsilon \ll 1$. Plugging it in the resource equation, it becomes $\epsilon
\frac{dR}{dt} = \left[ R (1-\beta R) - (1-\beta) RN \right]$. Since $\epsilon$
is very small, it can be a good approximation to set it to zero, so that the
resource equation becomes just an algebraic equation:

$$ R (1-\beta R) - (1-\beta) RN = 0 \therefore R = \frac{1 - (1-\beta)N}{\beta}
$$

Substituting it back into the differential equation for the consumer, we find

$$ \frac{dN}{dt} = N \left[\frac{1 - (1-\beta)N}{\beta} - 1\right]
   = \frac{1-\beta}{\beta} N (1-N) $$

We just recovered the logistic equation! Let's write the equation above in
dimensional form again:

$$ \frac{dN}{dt} = (ecK-d) N \left[ 1-\frac{N}{\frac{r}{c}\left(
1-\frac{d}{ecK}\right)} \right] = r_C N \left(1-\frac{N}{K_C}\right)$$

So we find that the consumer-resource model we proposed, in the limit where the
resource growth rate is much faster than consumer mortality, leads to a logistic
growth for the consumer with maximum growth rate $r_C = (ecK-d)$ and carrying
capacity $K_C = \frac{r}{c}\left( 1-\frac{d}{ecK}\right)$. By approximating
$\epsilon=0$, we were able to derive a phenomenological model of population
regulation from a mechanistic model of consumption and limitation of resources,
and thus relate the phenomenological parameters of the logistic equation to
measurable quantities such as consumption and mortality rates, and efficiency of
conversion.

Still, the original consumer-resource model is not completely mechanistic, since
it assumes that the resource growth also saturates at a carrying capacity. This
is appropriate, as "mechanistic" and "phenomenological" are not two separate
categories, but are actually part of a continuum, with more mechanistic models
incorporating more explicit details of the processes involved.

We explore below how well this approximation behaves for different values of
parameters and initial conditions. We have several factors to take into account.
The first thing we should be careful is to look at it in the time-scale of the
consumer, that is much longer than that of the resource, and look at how
different the time-scales of consumer and resource are (that is, the value of
$\epsilon=\frac{d}{r}$).

In second place, regarding the stability analysis of the original system, we
could have damped oscillations that may influence how the solutions approach the
fixed point. This is determined by the sign of the discriminant: if it is
positive, perturbations around the fixed point do not overshoot the fixed point,
otherwise they oscillate towards it. The condition for the absence of
oscillation is:

$$ T^2 - 4\Delta = \left(\frac{r}{K}\right)^2 - 4 dr\left(1 -
\frac{d}{ecK}\right) > 0 $$

Finally, initial conditions may be important: if we use both models with the
same initial values for the consumer, the first model will behave differently
depending on the initial population of the resource. If the initial consumer
population is small (compared to $N^*$), we expect that the resource population
starts near (or at) the carrying capacity, which should yield dynamics closer to
that of the logistic model.

First, we look at the simplest case: $r$ is $1000$ times larger than $d$
($\epsilon = 0.001$), the dynamics of the consumer-resource model are asymptotic
(no oscillatory behavior: $T^2-4\Delta > 0$).


    %pylab inline
    # larger plots and fonts, please
    pylab.rcParams['figure.figsize'] = (10.0, 6.0)
    pylab.rcParams['font.size'] = 12
    from scipy.integrate import odeint
    ion()
    
    def LVK(y, t, r, K, c, e, d):
        """Lotka-Volterra equations with a resource carrying capacity."""
        return array([ r * y[0] * (1-y[0]/K) - c*y[0]*y[1],
                       y[1] * (e*c*y[0] -d) ])
    
    def logist(y, t, rC, KC):
        """The logistic equation."""
        return rC*y*(1-y/KC)

    Populating the interactive namespace from numpy and matplotlib



    # parameters of C-R model
    r = 10.
    K = 10.
    c = 0.1
    e = 0.1
    d = 0.01
    # corresponding parameters of single-species model
    rC = e*c*K-d
    KC = r/c * (1-d/(e*c*K))
    
    t = arange(0, 200, 0.1)
    y0 = [K, .1]
    # we integrate the C-R model
    y = odeint(LVK, y0, t, (r, K, c, e, d))
    # and the logistic model
    yC = odeint(logist, y0[1], t, (rC, KC))
    
    # the time-scales ratio epsilon
    print('epsilon: ', d/r)
    # check the stability, if the discriminant T^2-4 Delta is positive,
    # the solutions do not approach the fixed point oscillating
    print('discriminant: ', (r/K)**2 - 4*d*r*(1-d/(e*c*K)))
    print('initial condition: ', y0)
    
    plot(t, y[:,0], label='resource')
    plot(t, y[:,1], label='R-C model')
    plot(t, yC, label='1-species model')
    xlabel('time')
    ylabel('population')
    ylim((0, KC*1.1))
    legend(loc='best');

    epsilon:  0.001
    discriminant:  0.6399999999999999
    initial condition:  [10.0, 0.1]



![png](From%20C-R%20to%20single-species%20dynamics_files/From%20C-R%20to%20single-species%20dynamics_4_1.png)


The agreement is quite good! Now we make $r$ smaller and $d$ larger, reducing
$\epsilon$ from $1/1000$ to just $1/10$. Notice that, as we do that, we also
make $T^2-4\Delta$ negative.


    # parameters of C-R model
    r = .5
    K = 10.
    c = 0.1
    e = 0.1
    d = 0.05
    # corresponding parameters of single-species model
    rC = e*c*K-d
    KC = r/c * (1-d/(e*c*K))
    
    t = arange(0, 400, 0.1)
    y0 = [K, .1]
    # we integrate the C-R model
    y = odeint(LVK, y0, t, (r, K, c, e, d))
    # and the logistic model
    yC = odeint(logist, y0[1], t, (rC, KC))
    
    # the time-scales ratio epsilon
    print('epsilon: ', d/r)
    # check the stability, if the discriminant T^2-4 Delta is positive,
    # the solutions do not approach the fixed point oscillating
    print('discriminant: ', (r/K)**2 - 4*d*r*(1-d/(e*c*K)))
    
    plot(t, y[:,0], label='resource')
    plot(t, y[:,1], label='R-C model')
    plot(t, yC, label='1-species model')
    xlabel('time')
    ylabel('population')
    ylim((0, KC*1.2))
    legend(loc='best');

    epsilon:  0.1
    discriminant:  -0.0475



![png](From%20C-R%20to%20single-species%20dynamics_files/From%20C-R%20to%20single-species%20dynamics_6_1.png)


We see that, although the final solution is the same (as it should always be!),
the relative difference in the dynamics in the transient increased. Now it is
your time to explore: try changing the initial conditions and parameter values
and see when the approximation breaks badly!

## References

The Consumer-Resource model was proposed by MacArthur a long time ago, and he
then already pointed out that by substituting the value of the resource
population at equilibrium back into the consumer equation, we recover the
logistic equation. Schaffer later generalized this approach to more general
forms. The recent paper by Reynolds and Brassil (which begins with the approach
we followed here) treats the same question from the point of view of separating
time scales, a topic that has been receiving some attention in the recent
theoretical literature.

* Robert H. MacArthur (1972) Geographical Ecology: patterns in the distribution
of species
* [Schaffer 1981](http://www.esajournals.org/doi/abs/10.2307/2937321)
* [Reynolds and Brassil
2013](https://www.sciencedirect.com/science/article/pii/S0022519313003998)
