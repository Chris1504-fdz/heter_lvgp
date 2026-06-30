function y = f_fun(x,t)
if t == 1 || t == 2
    y = sin(2*pi*x*t);
else
    y = x.^(t-1);
end