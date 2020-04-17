def indToXY(id, width, height):
    """Index to (x,y) location"""
    # Observation map is y-major coordinate
    x, y = id % width, id // width
    return [x, y]


def XYToInd(location, width, height):
    """Location (x,y) to index"""
    return location[1] * width + location[0]
