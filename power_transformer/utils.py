def duval_triangle(ch4_relative, c2h4_relative, c2h2_relative):
    """
    Determining the expected health state of the power transformer, according to the Duval triangle.
    """

    if ch4_relative >= 98:
        status = "PD"
    elif ch4_relative < 98 and c2h4_relative < 20 and c2h2_relative < 4:
        status = "T1"
    elif 50 > c2h4_relative >= 20 and c2h2_relative < 4:
        status = "T2"
    elif c2h4_relative >= 50 and c2h2_relative < 15:
        status = "T3"
    elif c2h4_relative < 50 and 13 > c2h2_relative >= 4:
        status = "DT"
    elif 40 <= c2h4_relative < 50 and 29 > c2h2_relative >= 13:
        status = "DT"
    elif 50 <= c2h4_relative and 29 > c2h2_relative >= 15:
        status = "DT"
    elif 23 < c2h4_relative and c2h2_relative >= 13:
        status = "D1"
    elif 23 <= c2h4_relative and c2h2_relative >= 29:
        status = "D2"
    elif 23 <= c2h4_relative < 40 and 29 > c2h2_relative >= 13:
        status = "D2"
    else:
        status = "other"

    return status
