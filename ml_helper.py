import numpy as np
from bcolors import Bcolors as bc
from sklearn import metrics
import re
import math
####
# MATRIX PRINTING
####
REX = re.compile('\\033\[\d+m')


# utility function to print a line of a certain length per cell
# lng will be the number of 'cell'
# L will be the width of a cell
def print_line_matrix(lng, L=8, f=None):
    print('-' * ((L+1) * (lng) + 1), file=f)


# format a string so it fits in a cell
# cutted to L characters (L-1 +'\' actually)
# Numbers have thousands separator if possible
# string are centered.
# It's possible to right aligne numbers but I don't like it
def format_string(ele, L=8):
    ele = str(ele)
    colors = REX.findall(ele)
    value = sorted(REX.split(ele))[-1]
    if value.replace('.', '').isdigit():
        if value.isdigit():
            f_value = int(value)
        else:
            f_value = float(value)
        tmp = '{:,}'.format(f_value).replace(',', ' ')
        if len(tmp) < L:
            value = tmp
    if len(value) > L:
        value = value[:L-1]+'\\'
    value = value[:L].center(L)
    return ''.join(colors[:-1])+value+''.join(colors[-1:])


# function to format the row of a matrix
# r are the different cell
# L is the width for a cell
def format_row(r, L=8):
    return '|' + '|'.join([format_string(i, L) for i in r]) + '|'


# print a 2d array based on a layout
# each cell will have L characters
# can have color code
def print_matrix(layout, L=8, f=None):
    print_line_matrix(len(layout[0]), L, f)
    for i in range(len(layout)):
        print(format_row(layout[i], L), file=f)
        len_l = len(layout[i])
        if i + 1 < len(layout):
            len_l = max(len(layout[i+1]), len_l)
        print_line_matrix(len_l, L, f)


def dvd(a, b):
    if b == 0 or b == float('inf') or math.isnan(b):
        return float('inf')
    else:
        return a/b


def get_roc_auc_score(predicted, label):
    res = {}
    try:
        res['auc'] = metrics.roc_auc_score(label, predicted)
    except:  # bare except bad, todo (yeah I know)
        res['auc'] = 'nan'
    return res


def get_cohen_kappa(predicted, label):
    res = {}
    res['cks'] = metrics.cohen_kappa_score(predicted, label)
    return res


def get_matthews_correlation(predicted, label):
    res = {}
    res['mcc'] = metrics.matthews_corrcoef(predicted, label)
    return res


def get_balanced_accuracy(predicted, label):
    res = {}
    res['bas'] = metrics.balanced_accuracy_score(predicted, label)
    return res


# get multiple values out of a confusion matrix
# recall (tpr)
# precition (ppv)
def get_score_main(matrix):
    res = {}
    res['ppv'] = dvd(matrix[0][0], (matrix[:, 0].sum()))
    res['tpr'] = dvd(matrix[0][0], matrix[0].sum())
    return res


# get multiple values out of a confusion matrix
# recall (tpr)
# precition (ppv)
def get_score_main_ext(matrix):
    res = {}
    res['ppv'] = dvd(matrix[0][0], (matrix[:, 0].sum()))
    res['npv'] = dvd(matrix[1][1], matrix[:, 1].sum())
    res['tpr'] = dvd(matrix[0][0], matrix[0].sum())
    res['tnr'] = dvd(matrix[1][1], matrix[1].sum())
    return res


def get_score_predicted_1(matrix):
    res = {}
    res['tpr'] = dvd(matrix[0][0], matrix[0].sum())
    res['fpr'] = dvd(matrix[1][0], matrix[1].sum())
    return res


def get_score_predicted_2(matrix):
    res = {}
    res['fnr'] = dvd(matrix[0][1], matrix[0].sum())
    res['tnr'] = dvd(matrix[1][1], matrix[1].sum())
    return res


# get multiple values out of a confusion matrix
# recall (tpr)
# fall-out  (fpr)
# miss rate (fnr)
# specificity (tnr)
def get_score_predicted(matrix):
    res = {}
    res = {**res, **get_score_predicted_1(matrix)}
    res = {**res, **get_score_predicted_2(matrix)}
    return res


def get_score_label_1(matrix):
    res = {}
    res['ppv'] = dvd(matrix[0][0], matrix[:, 0].sum())
    res['for'] = dvd(matrix[0][1], matrix[:, 1].sum())
    return res


def get_score_label_2(matrix):
    res = {}
    res['fdr'] = dvd(matrix[1][0], matrix[:, 0].sum())
    res['npv'] = dvd(matrix[1][1], matrix[:, 1].sum())
    return res


# get multiple values out of a confusion matrix
# precision (ppv)
# false discovery rate  (fdr)
# false omission rate (for)
# negative predictive value (npv)
def get_score_label(matrix):
    res = {}
    res = {**res, **get_score_label_1(matrix)}
    res = {**res, **get_score_label_2(matrix)}
    return res


def get_accuracy(matrix):
    res = {}
    res['acc'] = dvd(sum(matrix.diagonal()), matrix.sum())
    return res


def get_prevalence(matrix):
    res = {}
    res['pre'] = dvd(matrix[0].sum(), matrix.sum())
    return res


# get multiple values out of a confusion matrix
# accuracy (acc)
# prevalence (pre)
def get_score_total(matrix):
    res = {}
    res = {**res, **get_accuracy(matrix)}
    res = {**res, **get_prevalence(matrix)}
    return res


# get multiple values out of scores of a classification
# positive likelihood ratio (lr+)
# negative likelihood ratio (lr-)
def get_score_ratio(score):
    res = {}
    res['lr+'] = dvd(score['tpr'], score['fpr'])
    res['lr-'] = dvd(score['fnr'], score['tnr'])
    return res


# get the f1 value  out of scores of a classification
def get_score_f1(score):
    res = {}
    denom = (score['ppv'] + score['tpr'])
    if denom == 0:
        res['f_1'] = 0
    elif denom == float('inf'):
        res['f_1'] = 0
    else:
        res['f_1'] = 2.0 * dvd(score['ppv'] * score['tpr'], denom)
    return res


# get multiple values out of scores of a classification
# f1 score (f_1)
# diagnostic odds ratio (dor)
def get_score_about_score(score):
    res = get_score_f1(score)
    res['dor'] = dvd(score['lr+'], score['lr-'])
    return res


# get all values out of a confusion matrix
# recall (tpr)
# fall-out  (fpr)
# miss rate (fnr)
# specificity (tnr)
# precision (ppv)
# false discovery rate  (fdr)
# false omission rate (for)
# negative predictive value (npv)
# accuracy (acc)
# prevalence (pre)
# positive likelihood ratio (lr+)
# negative likelihood ratio (lr-)
# f1 score (f_1)
# diagnostic odds ratio (dor)
# area under the roc curve (auc)
def get_all_score(predicted, label, matrix):
    res = get_score_predicted(matrix)
    res = {**res, **get_score_label(matrix)}
    res = {**res, **get_score_total(matrix)}
    res = {**res, **get_score_ratio(res)}
    res = {**res, **get_score_about_score(res)}
    res = {**res, **get_roc_auc_score(predicted, label)}
    res = {**res, **get_cohen_kappa(predicted, label)}
    res = {**res, **get_matthews_correlation(predicted, label)}
    res = {**res, **get_balanced_accuracy(predicted, label)}
    return res


def get_score_read_test(predicted, label, matrix):
    res = get_score_predicted(matrix)
    res = {**res, **get_score_about_score(res)}
    res = {**res, **get_accuracy(matrix)}

# get multiple values out of a confusion matrix
# recall (tpr)
# precision (ppv)
# accuracy (acc)
# prevalence (pre)
# f1 score (f_1)
# area under the roc curve (auc)
def get_score_verbose_2(predicted, label, matrix):
    res = get_score_main_ext(matrix)
    res = {**res, **get_score_total(matrix)}
    res = {**res, **get_score_f1(res)}
    res = {**res, **get_roc_auc_score(predicted, label)}
    return res


####
# Utility
####
blue = ['ppv', 'tpr', 'auc', 'f_1', 'acc']
yellow = ['tnr', 'npv', 'lr+']


# add color to a layout for pretty printing
# color the true in green and false in red
# color the value in the array above (blue, yellow) in blue or yellow
def add_color_layout(layout):
    layout[1][1] = bc.LGREEN + str(layout[1][1]) + bc.NC
    layout[2][2] = bc.LGREEN + str(layout[2][2]) + bc.NC
    layout[1][2] = bc.LRED + str(layout[1][2]) + bc.NC
    layout[2][1] = bc.LRED + str(layout[2][1]) + bc.NC
    # this should be a function somewhere, to much copy paste
    for i in range(0, min(len(layout), 4)):
        for j in range(len(layout[i])):
            if (layout[i][j] in blue):
                layout[i][j] = bc.CYAN + layout[i][j] + bc.NC
                ii = i+1 if i % 2 == 0 else i-1
                layout[ii][j] = bc.LCYAN + layout[ii][j] + bc.NC
            elif (layout[i][j] in yellow):
                layout[i][j] = bc.YELLOW + layout[i][j] + bc.NC
                ii = i+1 if i % 2 == 0 else i-1
                layout[ii][j] = bc.LYELLOW + layout[ii][j] + bc.NC
    for i in range(4, len(layout)):
        for j in range(len(layout[i])):
            if (layout[i][j] in blue):
                layout[i][j] = bc.CYAN + layout[i][j] + bc.NC
                jj = j+1 if j % 2 == 0 else j-1
                layout[i][jj] = bc.LCYAN + layout[i][jj] + bc.NC
            elif (layout[i][j] in yellow):
                layout[i][j] = bc.YELLOW + layout[i][j] + bc.NC
                jj = j+1 if j % 2 == 0 else j-1
                layout[i][jj] = bc.LYELLOW + layout[i][jj] + bc.NC


# append a series of value to the end of the first lines of a 2d list
# ele: the 2d list of keys to add
# score: the dict to take the value from using the key
# layout: 2d list to append the value to
# inv: 1d array to specify the order (key, value) or (value, key), for each row
# ele: [['acc','auc'],['pre']]
# layout: [[a,b,c],[e,f,g],[h,i,j],[k,l,m]]
# inv: [0,1]
#
# layout = [[a,b,c,'acc','auc'],
#           [e,f,g,score['acc'],score['auc']],
#           [h,i,j,score['pre']],
#           [k,l,m,'pre']]
def append_layout_col(ele, score, layout, inv=None):
    if inv is None:
        inv = []
    inv.extend(np.zeros(len(ele) - len(inv), dtype=int))
    for i in range(len(ele)):
        layout[i*2+(inv[i] % 2)] += ele[i]
        layout[i*2+((1+inv[i]) % 2)] += [score[j] for j in ele[i]]


# append a series of value to the end of a 2d list
# ele: the 2d list of keys to add
# score: the dict to take the value from using the key
# layout: 2d list to append the value to
# inv: 1d array to specify the order (key, value) or (value, key), for each col
# ele: [['acc','auc'],['pre']]
# layout: [[a,b,c],[e,f,g]]
# inv: [0,1]
#
# layout = [[a,b,c],
#           [e,f,g],
#           ['acc', score['acc'], score['pre'], 'pre'],
#           [score['auc'],'auc']]
def append_layout_row(ele, score, layout, inv=None):
    if inv is None:
        inv = []
    inv.extend(np.zeros(len(ele[0]) - len(inv), dtype=int))
    to_print = [[k for i in range(len(ele[j]))
                 for k in
                 ([ele[j][i], score[ele[j][i]]]
                 if inv[i] == 0 else
                 [score[ele[j][i]], ele[j][i]])]
                for j in range(len(ele))]
    layout.extend(to_print)


# clean a 2d array to make it ready for formatting
# round float and convert all element to string
def clean_layout(layout):
    layout = [[str(round(i, 3)) if isinstance(i, float) else str(i) for i in j]
              for j in layout]
    return layout


def append_verbose_3(layout, predicted, label, matrix):
    score = get_all_score(predicted, label, matrix)
    append_layout_col([['tpr', 'fnr', 'acc'],
                       ['fpr', 'tnr', 'pre']],
                      score, layout, inv=[0, 1])
    append_layout_row([['ppv', 'for', 'f_1'],
                       ['fdr', 'npv', 'auc'],
                       ['lr+', 'lr-', 'dor']],
                      score, layout, inv=[0, 1, 0])


def append_verbose_2(layout, predicted, label, matrix):
    score = get_score_verbose_2(predicted, label, matrix)
    append_layout_col([['tpr', 'acc'], ['f_1', 'pre']], score, layout)
    append_layout_row([['ppv', 'auc']], score, layout)


def append_verbose_1(layout, predicted, label, matrix):
    score = get_score_total(matrix)
    append_layout_col([['acc'], ['pre']], score, layout)


def append_verbose_0(layout, predicted, label, matrix):
    layout.append(['total', matrix[:, 0].sum(),
                   matrix[:, 1].sum(), matrix.sum()])
    layout[0].append('total')
    layout[1].append(matrix[0].sum())
    layout[2].append(matrix[1].sum())


# print a comparison of the result of a classification
# label vs predicted
# it will print a confusion matrix
# verbose: how much measure are to be displayed (0,1,2,3)
# color: put color in
# L: cell width
def compare_class(predicted, label, verbose=1, color=True, L=8, unique_l=None,
                  f=None):
    if unique_l is None:
        unique_l = np.unique(label)[::-1]
    matrix = metrics.confusion_matrix(
        label, predicted, labels=unique_l)
    layout = [['lb\pr', *unique_l],
              [unique_l[0], *matrix[0]],
              [unique_l[1], *matrix[1]]]
    if (verbose > 0):
        append_verbose_0(layout, predicted, label, matrix)
        if (verbose == 1):
            append_verbose_1(layout, predicted, label, matrix)
        elif (verbose == 2):
            append_verbose_2(layout, predicted, label, matrix)
        elif (verbose == 3):
            append_verbose_3(layout, predicted, label, matrix)
    layout = clean_layout(layout)
    if color:
        add_color_layout(layout)
    print_matrix(layout, L, f)


def compare_per_patient(predicted, label, p_test, i_test,
                        verbose=1, color=True, L=8, unique_l=None):
    for i in p_test:
        compare_class(predicted[i_test == i], label[i_test == i],
                      verbose=verbose, color=color, L=L, unique_l=unique_l)
