"""
Single-variable Feynman equations testable by OUROBOROS.
Each entry: (name, expression_str, generator_fn, alphabet_size)
generator_fn(t) -> int  (discretized for MDL comparison)
"""
import math

FEYNMAN_1VAR = [
    # (name, true_expr, fn, n_points, alphabet_size)
    ("I.6.2",    "exp(-t^2/2)/sqrt(2pi)",  lambda t: int(round(math.exp(-(t/5)**2/2)*100)), 100, 102),
    ("I.9.18",   "F/(m*r)",                lambda t: int(round(1000/((t+1)))),               100, 1002),
    ("I.12.1",   "F=q*Ef",    lambda t: int(round(3*t+3)),      100, 302),
    ("I.12.2",   "q1*q2/(4pi*e*r^2)",      lambda t: int(round(1000/((t+1)**2))),            50,  1002),
    ("I.15.10",  "m0/sqrt(1-v^2)",         lambda t: int(round(100/math.sqrt(1-(t/100)**2+1e-6))), 90, 200),
    ("I.18.4",   "2t+3",      lambda t: int(round(2*t+3)),      100, 206),
    ("I.26.2",   "arcsin(n*sin(t))",       lambda t: int(round(math.asin(min(0.5*math.sin((t+1)*0.1),0.99))*100)), 100, 200),
    ("I.29.4",   "1/2*e*Ef^2",            lambda t: int(round(0.5*(t+1)**2)),               50,  1252),
    ("I.30.5",   "n*d*sin(t)",            lambda t: int(round(3*(t+1)*math.sin((t+1)*0.1))), 100, 200),
    ("I.34.8",   "q*Ef/m",               lambda t: int(round(5*(t+1))),                     100, 502),
    ("I.34.14",  "w0/(1-v)",             lambda t: int(round(10*(t+1))),                     100, 1002),
    ("I.34.27",  "h*w",       lambda t: int(round(3*t+3)),      100, 302),
    ("I.37.4",   "I1+I2+2*sqrt(I1*I2)",  lambda t: int(round((math.sqrt(t+1)+math.sqrt(t+2))**2)), 100, 200),
    ("I.43.31",  "mob*kb*T",             lambda t: int(round(2*(t+1))),                      100, 202),
    ("I.50.26",  "x+x*e*t",             lambda t: int(round((t+1)*math.exp(0.1*(t+1)))),    50,  300),
    ("II.2.42",  "kappa*(T2-T1)/d",     lambda t: int(round(5*(t+1))),                      100, 502),
    ("II.6.15a", "e*Ef/(m*w^2)",        lambda t: int(round(100/((t+1)**2))),               50,  102),
    ("II.11.3",  "n0*exp(-m*g*x/kb/T)", lambda t: int(round(100*math.exp(-0.05*(t+1)))),   50,  102),
    ("II.11.17", "n0*exp(-m*g*x/kb/T)", lambda t: int(round(50*math.exp(-0.1*(t+1)))),     50,  52),
    ("II.11.27", "n*alpha/(1-n*alpha/3)",lambda t: int(round(10*(t+1)/(1+0.1*(t+1)))),     50,  200),
    ("II.11.28", "1+n*alpha/(1-n*a/3)", lambda t: int(round(1+(t+1)*0.5)),                  50,  30),
    ("II.34.2",  "q*v/(2*pi*r)",        lambda t: int(round(10/(t+1))),                     50,  12),
    ("II.34.29a","q*h/(4*pi*m)",        lambda t: int(round(5*(t+1))),                      50,  252),
    ("III.4.32", "h*w/(exp(h*w/kb/T)-1)",lambda t: int(round(10/(math.exp(0.1*(t+1))-1+1e-6))), 50, 50),
    ("III.4.33", "h*w*exp(h*w/kb/T)/(exp(h*w/kb/T)-1)^2", lambda t: int(round(10*math.exp(0.1*(t+1))/(math.exp(0.1*(t+1))-1+1e-6)**2)), 40, 50),
    # Purely structural — easy wins
    ("linear_1", "3*t+2",     lambda t: int(round(3*t+2)),      100, 305),
    ("linear_2", "7*t",                 lambda t: int(round(7*(t+1))),                      100, 702),
    ("quad_1",   "t^2",                 lambda t: int((t+1)**2),                            30,  902),
    ("quad_2",   "2*t^2+1",             lambda t: int(2*(t+1)**2+1),                        30,  1802),
    ("log_1",    "10*log(t+1)",         lambda t: int(round(10*math.log(t+1+1))),           100, 42),
    ("sqrt_1",   "10*sqrt(t+1)",        lambda t: int(round(10*math.sqrt(t+1+1))),          100, 102),
    ("exp_1",    "100*exp(-t/10)",      lambda t: int(round(100*math.exp(-(t+1)/10))),      50,  102),
    ("inv_1",    "100/(t+1)",           lambda t: int(round(100/(t+1+1))),                  50,  52),
    ("inv2_1",   "100/(t+1)^2",        lambda t: int(round(100/(t+1+1)**2)),               30,  102),
]