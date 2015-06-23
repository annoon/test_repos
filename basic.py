def meh(x,y):
    if x>y:
        x=x-1
        print str(x) + ' ' + str(y)
        return meh(x,y)
    else:
        print 'done'
meh(10,1)