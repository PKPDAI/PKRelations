from scripts.others.extract_lang import analyse_all

text1 = "Itraconazole affected the pharmacokinetic parameters of S-fexofenadine more, and increased AUC(0," \
        "24 h) of S-fexofenadine and R-fexofenadine by 4.0-fold (95% CI of differences 2.8, 5.3; P < 0.001) and by " \
        "3.1-fold (95% CI of differences 2.2, 4.0; P = 0.014), respectively, and Ae(0,24 h) of S-fexofenadine and " \
        "R-fexofenadine by 3.6-fold (95% CI of differences 2.6, 4.5; P < 0.001) and by 2.9-fold (95% CI of " \
        "differences 2.1, 3.8; P < 0.001), respectively. "
text2 = "The volume of distribution of racemic primaquine was decreased by a median (95% CI) of 22.0% (2.24%-39.9%), " \
        "24.0% (15.0%-31.5%) and 25.7% (20.3%-31.1%) when co-administered with chloroquine, " \
        "dihydroartemisinin/piperaquine and pyronaridine/artesunate, respectively. "
text3 = "In healthy male volunteers, mean Cmax and AUC(0-10 h) of cabergoline increased to a similar degree during " \
        "co-administration of clarithromycin. "
text4 = "The clearance of mitoxantrone and etopside was decreased by 64% and 60%, respectively, when combined with " \
        "valspodar "
text5 = 'The peak plasma concentration of rosiglitazone was significantly decreased by rifampin (537.7 ng/mL versus ' \
        '362.3 ng/mL, P <.01). '
text6 = 'The apparent oral clearance of rosiglitazone increased about 3-fold after rifampin treatment (2.8 L/h versus ' \
        '8.5 L/h, P <.001). '
text7 = 'Ketoconazole increased the maximum observed plasma concentration (C(max)) and area under the plasma ' \
        'concentration time curve to the last sampling time, t (AUC(0-t)) of single-dose casopitant 2.7-fold and ' \
        '12-fold and increased the C(max) of 3-day casopitant 2.5-fold on day 1 and 2.9-fold on day 3, whereas AUC((' \
        '0-tau)) increased 4.3-fold on day 1 and 5.8-fold on day 3. '
text8 = 'Repeat-dose rifampin reduced the C(max) and AUC((0-t)) of casopitant 96% and 90%, respectively.'
text9 = 'Itraconazole decreased plasma clearance (Cl) and increased the area under the plasma concentration-time ' \
        'curve (AUC 0-infinity) of intravenous oxycodone by 32 and 51%, respectively (P<0.001) and increased the AUC(' \
        '0-infinity) of orally administrated oxycodone by 144% (P<0.001). '
text10 = "Midazolam increased the AUC of amoxicillin"
text11 = "Midazolam's AUC was increased by amoxicillin"
text12 = "Midazolam's AUC was increased through amoxicillin administration"
text13 = "The clearance of mitoxantrone and etopside was decreased by 64% and 60%, respectively, when combined with " \
         "valspodar "
text14 = "The bioavailability of oral ondansetron was reduced from 60% to 40% (P<.01) by rifampin"
text15 = "The volume of distribution of racemic primaquine was decreased by a median (95% CI) of 22.0% (2.24%-39.9%), " \
         "24.0% (15.0%-31.5%) and 25.7% (20.3%-31.1%) when co-administered with chloroquine, " \
         "dihydroartemisinin/piperaquine and pyronaridine/artesunate, respectively. "

assert analyse_all(text1) == [(['Itraconazole'], 'affected', ['AUC(0,24 h)'], ['S-fexofenadine', 'R-fexofenadine']),
                              (['Itraconazole'], 'increased', ['AUC(0,24 h)'], ['S-fexofenadine', 'R-fexofenadine'])]
assert analyse_all(text2) == [(
                              ['chloroquine', 'dihydroartemisinin/piperaquine', 'pyronaridine/artesunate'], 'decreased',
                              ['volume of distribution'], ['primaquine']), (['[]'], 'co-administered', [], [])]
assert analyse_all(text3) == [(['[]'], 'mean', ['Cmax', 'AUC(0-10 h)'], ['cabergoline']),
                              (['clarithromycin'], 'increased', ['Cmax', 'AUC(0-10 h)'], ['cabergoline'])]
assert analyse_all(text4) == [(['valspodar'], 'decreased', ['clearance'], ['mitoxantrone', 'etopside']),
                              (['[]'], 'combined', [], [])]
assert analyse_all(text5) == [(['rifampin'], 'decreased', ['peak plasma concentration'], ['rosiglitazone'])]
assert analyse_all(text6) == [(['rifampin'], 'increased', ['apparent oral clearance'], ['rosiglitazone'])]
assert analyse_all(text7) == [(['Ketoconazole'], 'increased', ['maximum observed plasma concentration (C(max))',
                                                               'area under the plasma concentration time curve to the last sampling time, t (AUC(0-t))'],
                               ['casopitant']), (['Ketoconazole'], 'increased', ['C(max)'], ['casopitant']),
                              (['Ketoconazole'], 'increased', ['AUC((0-tau)'], ['casopitant'])]
assert analyse_all(text8) == [(['Repeat-dose rifampin'], 'reduced', ['C(max)', 'AUC((0-t))'], ['casopitant'])]
# careful this one first got wrong if it breaks (because it gets the right one) here that'd be great
assert analyse_all(text9) == [(['Itraconazole'], 'decreased', ['plasma clearance (Cl)'], []), (
['Itraconazole'], 'increased', ['area under the plasma concentration-time curve (AUC 0-infinity)'], ['oxycodone']),
                              (['Itraconazole'], 'increased', ['AUC(0-infinity)'], ['oxycodone']),
                              (['[]'], 'administrated', [], [])]
assert analyse_all(text10) == [(['Midazolam'], 'increased', ['AUC'], ['amoxicillin'])]
assert analyse_all(text11) == [(['amoxicillin'], 'increased', ['AUC'], ['Midazolam'])]
assert analyse_all(text12) == [(['amoxicillin'], 'increased', ['AUC'], ['Midazolam'])]
assert analyse_all(text13) == [(['valspodar'], 'decreased', ['clearance'], ['mitoxantrone', 'etopside']),
                               (['[]'], 'combined', [], [])]
assert analyse_all(text14) == [(['rifampin'], 'reduced', ['bioavailability'], ['ondansetron'])]
assert analyse_all(text15) == [(['chloroquine', 'dihydroartemisinin/piperaquine', 'pyronaridine/artesunate'],
                                'decreased', ['volume of distribution'], ['primaquine']),
                               (['[]'], 'co-administered', [], [])]
