def indToXY(id, width, height):
    """Index to (x,y) location"""
    # Observation map is y-major coordinate
    y, x = id % width, id // width
    return [x, y]


def XYToInd(location, width, height):
    """Location (x,y) to index"""
    return location[0] * width + location[1]
