/* Original XKCP theta schedule (CC0, Ronny Van Keer). All five
 * column parities must remain intact until the five D values are formed.
 */
#define KeccakP_ThetaRhoPiChiIota0(_L1, _L2, _L3, _L4, _L5, _rc)       \
    _Ba = XOR5(_ba, _ga, _ka, _ma, _sa);                               \
    _Be = XOR5(_be, _ge, _ke, _me, _se);                               \
    _Bi = XOR5(_bi, _gi, _ki, _mi, _si);                               \
    _Bo = XOR5(_bo, _go, _ko, _mo, _so);                               \
    _Bu = XOR5(_bu, _gu, _ku, _mu, _su);                               \
    _Da = XOR(ROL(_Be, 1), _Bu);                                       \
    _De = XOR(ROL(_Bi, 1), _Ba);                                       \
    _Di = XOR(ROL(_Bo, 1), _Be);                                       \
    _Do = XOR(ROL(_Bu, 1), _Bi);                                       \
    _Du = XOR(ROL(_Ba, 1), _Bo);                                       \
    KeccakP_ThetaRhoPiChi(_L1, _L2, _L3, _L4, _L5, _Ba, _Be, _Bi, _Bo, \
                          _Bu, 0, 44, 43, 21, 14);                     \
    _L1 = XOR(_L1, _rc);
