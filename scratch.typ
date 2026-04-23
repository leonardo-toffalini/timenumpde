= ASD

Plane wave
$
  (n delta, k h) -> e^(i (omega n delta - xi k h))
$

== (a1)
$
  u^(n+1) = u^(n-1) - mu dot D_0 u^n
$

_Solution:_

$
  u^(n+1)_k = u^(n-1)_k - (a delta)/h  (u^n_(k+1) - u^n_(k-1))
$
$
  u^n_k dot e^(i omega delta) &= u^n_k dot e^(-i omega delta) - (a delta)/h (u^n_k dot e^(-i xi h) - u^n_k dot e^(i xi h)) \
  &= u^n_k (e^(i omega delta) + (a delta)/h (e^(i xi h) - e^(- i xi h))) \
  &= u^n_k (e^(i omega delta) + (a delta)/h 2 i sin(xi h)) \
$

$
  2 i sin (omega delta) &= - mu dot (- 2 i sin(xi h)) \
  sin(omega delta)/(sin (xi h)) &= mu
$

== (a2)
$
  u^(n+1) = u^n - mu dot D_(-) u^n
$

_Solution:_
$
  e^(i omega delta) &= 1 - mu (1 - e^(i xi h)) \
  &= 1 - mu + e^( i xi h)
$

$
  cos (omega delta) + i sin (omega delta) = 1 - mu + mu(cos (xi h) + i sin(xi h))
$

Complex part, same as in (a1):
$
  sin (omega delta) = mu sin (xi h)
$

Real part:
$
  cos(omega delta) = 1 - mu + mu cos(xi h)
$

== (a3)
$
  u^(n+1) = u^n - mu dot D_0 u^(n+1)
$

== (a4)
$
  u^(n+1) = u^n - mu dot D_0 (u^n + u^(n+1))/2
$

_Solition:_

$
  e^(i omega delta) = 1 - mu (e^(-i xi h))
$


