#import "@preview/intextual:0.1.0": flushr, intertext-rule
#import "@preview/theorion:0.4.1": *
#import cosmos.default: *
#show: show-theorion
#show: intertext-rule

#set text(size: 12pt, font: "New Computer Modern Math")
#set par(justify: true, first-line-indent: 1em)
#set page(margin: 4em, numbering: "1")
// #set math.equation(numbering: "(1)")

#let uu = $bold(u)$
#let ff = $bold(f)$
#let vv = $bold(v)$
#let ww = $bold(w)$
#let FF = $bold(F)$

#let numbered_eq(content) = math.equation(
    block: true,
    numbering: "(1)",
    content,
)

#let bracket(a, b) = $angle.l #a, #b angle.r$

// Title
#align(center)[
  #text(size: 1.2em)[= Numerical method for solving the \ 1 dimensional wave equation]

  #v(0.5em)
  #text(size: 1.25em)[Leonardo Toffalini]
]
#v(2em)

#set heading(numbering: "1.1")

= Introduction
The wave equation in 1 dimension on the interval $[0, pi/2]$ looks as follows

$
  cases(
    partial_(t t) u(t, x) &= partial_(x x) u(t, x) + f flushr((x in (0, pi/2)\, quad t in (0, 1))),
    partial_t u(0, x) &= 0 flushr((x in (0, pi/2))),
    u(0, x) &= cos x flushr((x in (0, pi/2))),
    partial_x u(t, 0) &= 0 flushr((t in (0, 1))),
    u(t, pi/2) &= t^2 flushr((t in (0, 1)))
  )
$
where $f equiv 2$.

The analytic solution is $u(t, x) = t^2 + cos t cos x$.

== Numerical method

We are going to rely on the second order approximation of $partial_(x x)$ and
$partial_(t t)$ which look as follows
$
  partial_(x x) u(t, x) &= 1/h_x^2 (u(t, x+h_x) - 2 u(t, x) + u(t, x - h_x)) + O(h_x^2) \
  partial_(t t) u(t, x) &= 1/delta^2 (u(t + delta, x) - 2 u(t, x) + u(t - delta, x)) + O(delta^2),
$
where we discretize in space with $h_x$ and $delta$ in time.

With the above finite difference schemes we can rewrite the exact equation into
a numerical procedure
$
  1/delta^2 (u(t + delta, x) - 2 u(t, x) + u(t - delta, x)) &= 1/h_x^2 (u(t, x+h_x) - 2 u(t, x) + u(t, x - h_x)) + f.
$

Rearranging the above equation to have only $u(t + delta, x)$ on the right hand
side we get
$
  u(t + delta, x) = 2 u(t, x) - u(t - delta, x) + delta^2/h_x^2 (u(t, x+ h_x) - 2 u(t, x) + u(t, x - h_x)) + delta^2 f.
$

Let us define $t_n := n delta$ and $x_k := k h_x$, with these $u_k^n := u(t_n,
x_k)$. Using these notations we can simplify to
#numbered_eq($
  u_k^(n+1) = 2 u_k^n - u_k^(n-1) + delta^2/h_x^2 (u_(k+1)^n - 2 u_k^n + u_(k-1)^n) + delta^2 f.
$)<eq:num_method>

Note that the error terms collect to form $delta^2 (O(delta^2) + O(h_x^2))$.

== Handling the boundary conditions
The easiest boundary condition to handle is $u(0, x) = cos x$, which just means that we initialize $u_k^0 = cos k h_x$.

Handling the boundary condition $u(t, pi/2) = t^2$ is just as trivial, we just
set $u_(N_x-1)^n = (n delta)^2$ in each time step, where $N_x$ is the number of
grid points in the space discretization.

Next, we handle $partial_t u(0, x) = 0$. Notice, that in @eq:num_method to get
$u_k^(t+1)$ we need $u_k^n$ and $u_k^(n-1)$, which means that we need $u_k^0$
and $u_k^1$ as initial values to start our numerical method. To combat this,
let us write out the second order Taylor approximation of $u(t, x)$ around
$t=0$:
$
  u(delta, x) = u(0, x) + delta overbrace(partial_t u(0, x), 0) + delta^2 partial_(t t) u(0, x) + O(delta^3).
$

We can see that the first order term vanishes due to the Neumann boundary
condition, and by the problem statement we know that
$
  partial_(t t) u(0, x) = partial_(x x) u(0, x) + f,
$
where we already have an approximation for $partial_(x x) u(0, x)$. Thus we
conclude that to initialize $u_k^1$ we can use the following formula
$
  u_k^1 = u_k^0 + delta^2/h_x^2 (u_(k+1)^0 - 2 u_k^0 + u_(k-1)^0) + delta^2 f.
$

The only boundary condition left to handle is $partial_x u(t, 0) = 0$. Notice
that for $u_0^(n+1)$ we would need $u_(-1)^n$ alongside $u_1^n$ and $u_0^n$. We
can instead use the Neumann boundary condition $partial_x u(t, 0) = 0$ to solve
this problem.

Let us write out the central finite difference of $partial_x$ to get some
equation for $u_(-1)^n$
$
  partial_x u_0^n approx 1/(2 h_x) (u_1^n - u_(-1)^n).
$

Since we know that $partial_x u_0^n = 0$ we have that $u_1^n = u_(-1)^n$ for
all time steps. With this trick we can resolve the previous problem by giving a
modified formula for the left end of the space interval
$
  partial_(x x) u_0^(n+1) &approx 1/h_x^2 (u_1^n - 2 u_0^n + u_(-1)^n) \
  &= 1/h_x^2 (u_1^n -2 u_0^n + u_1^n) \
  &= 2/h_x^2 (u_1^n - u_0^n).
$

In conclusion, we have handled the left and right ends of the space interval,
the left side was handled by $partial_x u(t, 0) = 0$, while the right side was
handled by $u(t, pi/2) = t^2$.

We have also handled the start of the time frame, where we needed two initial
values, one of which was provided as is $u(0, x) = cos x$, and the second we
figured out how to compute from the Neumann condition $partial_t u(0, x) = 0$.


