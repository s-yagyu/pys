# coding: utf-8
"""
fowler_func.py

Calculating  pys intensity from Fowler model

"""
import numpy as np
import matplotlib.pyplot as plt


def my_round(val, digit=0):
    """
    小数の四捨五入の関数

    Parameters
    ----------
    val: float
        四捨五入する数
    digit:　int , default 0
        四捨五入する桁
    Returns
    -------

    Examples
    --------
    >>>f=4.3499999999999996
    >>>g=my_round(f, 2)
    >>> print(g,type(g))
    4.35 <class 'float'>

    >>>g1=my_round(f,1)
    >>>print(g1,type(g1))
    4.4 <class 'float'>

    See Also
    ------
    Pythonで小数・整数を四捨五入するroundとDecimal.quantize
    https://note.nkmk.me/python-round-decimal-quantize/
    round()は一般的な四捨五入ではなく、偶数への丸めになる

    """
    p = 10 ** digit
    return (val * p * 2 + 1) // 2 / p

def x_lists(start, end, step, array='True'):
    """
    make ramp array
    
    Parameters
    ----------
    start : float
    end : float
    step : float
    array : bool
        if array == 'True' array
        else list

    Returns
    -------
    if array == 'True'
        array
    else
        list

    Note
    ------
    小数点以下の桁数を調整
    Stepを文字列に変え、小数点以下の文字数をカウント
    ランプ配列を作成

    xdata=x_axsis(4,6.2,0)リスト内包表現で書くと、この際はレンジ関数
    xdata = [3.5+x*0.01 for x in range(0,271,1)]

    """
    num = (my_round(((end - start) / (step)), 1) + 1)
    inum = int(num)
    step_str = str(step).split('.', -1)
    digit = len(step_str[-1])

    listx = [start + x * step for x in range(0, inum, 1)]
    xlist = [my_round(x, digit) for x in listx]

    if array == 'True':
        return np.array(xlist)

    else:
        return xlist

#pys function
@np.vectorize
def pys(x, Ip, T, Nor, Bg):
    '''
    pys calculation
    :param x: list of energy  or value
    :param Ip:
    :param T:
    :param Nor:
    :param Bg:
    :return: pys calculation

    example:
    -----
    >>>p=pys(np.array([4,4.2,4.3,4.5]),4.3,300,1,1)
    >>>print(p,type(p))
    [ 1.00000912  1.02078619  1.81083333 32.5716379 ] <class 'numpy.ndarray'>
    >>>p=pys(4.2,4.3,300,1,1)
    >>>print(p,type(p))
    1.020786189180001 <class 'numpy.ndarray'>

    Note:
    -----
    numpyのVectorizeを利用すると、配列を受け付けるようになる

    '''
    u = (x - Ip) / (T * 8.6171e-5)
    if u <= 0:
        f = Nor * (np.exp(u) - (np.exp(2 * u) / (2 * 2)) + (np.exp(3 * u) / (3 * 3))
                   - (np.exp(4 * u) / (4 * 4)) + (np.exp(5 * u) / (5 * 5)) - (np.exp(6 * u) / (6 * 6))) + Bg

    else:
        f = Nor * (np.pi * np.pi / 6 + (1 / 2) * u * u - (np.exp(-u)) + (np.exp(-2 * u) / (2 * 2))
                   - (np.exp(-3 * u) / (3 * 3)) + (np.exp(-4 * u) / (4 * 4)) - (np.exp(-5 * u) / (5 * 5))
                   + (np.exp(-6 * u) / (6 * 6))) + Bg

    return f

#sqrt PYS
@np.vectorize
def spys(x, Ip, T, Nor, Bg):
    """
    sqrt PYS
    
    """

    spys = np.sqrt(pys(x, Ip, T, Nor, Bg))
    
    return spys



### SPYS data creation
def SPYS_data(xdata,nor_list,ip_list,temp_list,bg_list):
    """
    
    Sqrt PYS data making
    all input data is list

    :param xdata: list
    :param nor_list:list
    :param ip_list:
    :param temp_list:
    :param bg_list:

    :return:
    XL : spectra
    yL : Ip
    """

    XL = []
    yL = []
    for Nor in nor_list:

        for Ip in ip_list:

            for Bg in bg_list:

                for T in temp_list:
                    spt = spys(x=xdata, Ip=Ip, T=T, Nor=Nor, Bg=Bg)
                    XL.append(spt)
                    yL.append(Ip)

    return XL, yL


# Maxcoutで制限,All paramaters return
def SPYS_data_ris_max_all(xdata, nor_list, ip_list, temp_list, bg_list, max_count=150, diff_count=7):
    '''
    理研計器ACシリーズでは、20000カウント程度で検出器が飽和する。
    その平方根(spysを計算するため)140（設定では150）以上のカウントのものはデータに加えない。
    diff_countで最小値と最大値のカウントの差が7以下は加えない。

    Sqrt PYS data making
    all input data is list

    :param xdata:
    :param nor_list:
    :param ip_list:
    :param temp_list:
    :param bg_list:
    :param max_count:maxcountの設定
    :param diff_count: difference of maxcount and mincount
    :return:XL, yL, yL_all
    '''

    XL = []
    yL = []
    yL_all = []


    for Nor in nor_list:

        for Ip in ip_list:

            for Bg in bg_list:

                for T in temp_list:
                    spt = spys(x=xdata, Ip=Ip, T=T, Nor=Nor, Bg=Bg)

                    if max_count >= np.max(spt) and diff_count <= np.max(spt) - np.min(spt):
                        XL.append(spt)
                        yL.append(Ip)
                        yL_all.append(np.array([Ip, T, Nor, Bg]))

    return XL, yL, yL_all