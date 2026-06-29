Vectorized means not using loops, NumPy itself does everything element wise directly. Be caredful and convert all of them to NumPy array and then do the calculation.

```
import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    param = np.array(param)
    grad = np.array(grad)
    m = np.array(m)
    v = np.array(v)
    m_t = (beta1*m)+ (1-beta1)*grad
    v_t = (beta2*v)+ (1-beta2)*(grad**2)
    m_hat = m_t/(1-beta1**t)
    v_hat = v_t/(1-beta2**t)
    par = param-lr*m_hat/(np.sqrt(v_hat)+eps)
    return par,m_t,v_t
```