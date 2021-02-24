
def __main(s):
    return "void main() { %s; }" % s

def default_raw():
    return __main("emitGrayscale(toNormalized(getDataValue()))")

def red_alpha():
    return __main("emitRGBA(vec4(1, 0, 0, toNormalized(getDataValue())))")

def green_alpha():
    return __main("emitRGBA(vec4(0, 1, 0, toNormalized(getDataValue())))")

def blue_alpha():
    return __main("emitRGBA(vec4(0, 0, 1, toNormalized(getDataValue())))")

def red():
    return __main("emitRGB(vec3(toNormalized(getDataValue()), 0, 0))")

def green():
    return __main("emitRGB(vec3(0, toNormalized(getDataValue()), 0))")

def blue():
    return __main("emitRGB(vec3(0, 0, toNormalized(getDataValue())))")

def rgb():
    return __main("emitRGB(vec3(toNormalized(getDataValue(0)),toNormalized(getDataValue(1)),toNormalized(getDataValue(2))))")

def jet():
    return __main("emitRGB(colormapJet(affinity))")