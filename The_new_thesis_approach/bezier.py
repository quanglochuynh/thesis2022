# _main.default.prototype.bezierPoint = function(a, b, c, d, t) {
#             _main.default._validateParameters('bezierPoint', arguments);

#             var adjustedT = 1 - t;
#             return (
#               Math.pow(adjustedT, 3) * a +
#               3 * Math.pow(adjustedT, 2) * t * b +
#               3 * adjustedT * Math.pow(t, 2) * c +
#               Math.pow(t, 3) * d
#             );
#           };

def bezier_point(a,b,c,d,t):
    aT = 1-t
    return (
        pow(aT,3) * a +
        3* pow(aT,2) * t * b + 
        3* aT* pow(t,2) * c + 
        pow(t,3) * d
    )