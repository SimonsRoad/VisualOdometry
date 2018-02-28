function[ex,ey] = getEllipse(x,y, w, h)
    t=-pi:0.01:pi;
    ex=x+w*cos(t);
    ey=y+h*sin(t);
end