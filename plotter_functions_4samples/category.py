"""
functions to divide in eta and energy bins

"""


from tqdm import tqdm as tqdm


def _divide_eta_categ(trackster_eta, n_eta_cat : int =8) -> int:
    """
    function to divide in eta bins
    """

    # boundary between low and high density: 2.15 eta
    # 5 high density eta bins: 1.65 - 1.75 - 1.85 - 1.95 - 2.05 - 2.15
    # 3 low density eta bins: 2.15 - 2.35 - 2.55 - 2.75
    # define list of eta bins boundaries
    bin_ed_list = [1.65, 1.75, 1.85, 1.95, 2.05, 2.15, 2.35, 2.55, 2.75]

    if trackster_eta >= bin_ed_list[0] and trackster_eta < bin_ed_list[1]:
        return 0
    elif trackster_eta >= bin_ed_list[1] and trackster_eta < bin_ed_list[2]:
        return 1
    elif trackster_eta >= bin_ed_list[2] and trackster_eta < bin_ed_list[3]:
        return 2
    elif trackster_eta >= bin_ed_list[3] and trackster_eta < bin_ed_list[4]:
        return 3
    elif trackster_eta >= bin_ed_list[4] and trackster_eta <= bin_ed_list[5]:
        return 4
    elif trackster_eta > bin_ed_list[5] and trackster_eta < bin_ed_list[6]:
        return 5
    elif trackster_eta >= bin_ed_list[6] and trackster_eta < bin_ed_list[7]:
        return 6
    elif trackster_eta >= bin_ed_list[7] and trackster_eta <= bin_ed_list[8]:
        return 7
    else:
        #print('ERROR: eta out of range')
        #raise ValueError('ERROR: eta out of range')
        return -1 # to be checked later, or ot will fill in the last categ


def _divide_en_categ(trackster_en, n_en_cat : int =9) -> int:
    """
    function to divide in energy bins
    """

    # boundary every 100 GeV
    # energy bins: 0 - 100 - 200 - 300 - 400 - 500 - 600 - 700 - 800 - inf
    bin_ed_list = [0, 100, 200, 300, 400, 500, 600, 700, 800]

    if trackster_en >= bin_ed_list[0] and trackster_en < bin_ed_list[1]:
        return 0
    elif trackster_en >= bin_ed_list[1] and trackster_en < bin_ed_list[2]:
        return 1
    elif trackster_en >= bin_ed_list[2] and trackster_en < bin_ed_list[3]:
        return 2
    elif trackster_en >= bin_ed_list[3] and trackster_en < bin_ed_list[4]:
        return 3
    elif trackster_en >= bin_ed_list[4] and trackster_en < bin_ed_list[5]:
        return 4
    elif trackster_en >= bin_ed_list[5] and trackster_en < bin_ed_list[6]:
        return 5
    elif trackster_en >= bin_ed_list[6] and trackster_en < bin_ed_list[7]:
        return 6
    elif trackster_en >= bin_ed_list[7] and trackster_en < bin_ed_list[8]:
        return 7
    elif trackster_en >= bin_ed_list[8]:
        return 8
    else:
        #print('ERROR: energy out of range')
        #return -1
        raise ValueError('ERROR: energy < 0')
