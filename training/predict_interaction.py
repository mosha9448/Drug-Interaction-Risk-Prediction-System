def classify_risk(prob):

    if prob < 0.3:
        return "LOW RISK"

    elif prob < 0.7:
        return "MEDIUM RISK"

    else:
        return "HIGH RISK"