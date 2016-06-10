/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * AbstNsoltCoefManipulator2d.c
 *
 * Code generation for function 'AbstNsoltCoefManipulator2d'
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "fcn_GradEvalSteps2d.h"
#include "AbstNsoltCoefManipulator2d.h"
#include "fcn_GradEvalSteps2d_emxutil.h"
#include "error1.h"
#include "eml_int_forloop_overflow_check.h"
#include "indexShapeCheck.h"
#include "fcn_GradEvalSteps2d_data.h"
#include "blas.h"

/* Variable Definitions */
static emlrtRSInfo sc_emlrtRSI = { 203, "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m"
};

static emlrtRSInfo tc_emlrtRSI = { 202, "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m"
};

static emlrtRSInfo vc_emlrtRSI = { 28, "reshape",
  "C:\\Program Files\\MATLAB\\R2016a\\toolbox\\eml\\lib\\matlab\\elmat\\reshape.m"
};

static emlrtRSInfo wc_emlrtRSI = { 58, "reshape",
  "C:\\Program Files\\MATLAB\\R2016a\\toolbox\\eml\\lib\\matlab\\elmat\\reshape.m"
};

static emlrtRSInfo xc_emlrtRSI = { 61, "reshape",
  "C:\\Program Files\\MATLAB\\R2016a\\toolbox\\eml\\lib\\matlab\\elmat\\reshape.m"
};

static emlrtRSInfo yc_emlrtRSI = { 108, "reshape",
  "C:\\Program Files\\MATLAB\\R2016a\\toolbox\\eml\\lib\\matlab\\elmat\\reshape.m"
};

static emlrtRSInfo be_emlrtRSI = { 235, "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m"
};

static emlrtRTEInfo n_emlrtRTEI = { 166, 18, "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m"
};

static emlrtRTEInfo o_emlrtRTEI = { 171, 13, "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m"
};

static emlrtRTEInfo p_emlrtRTEI = { 186, 13, "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m"
};

static emlrtRTEInfo q_emlrtRTEI = { 145, 18, "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m"
};

static emlrtRTEInfo s_emlrtRTEI = { 197, 26, "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m"
};

static emlrtRTEInfo ab_emlrtRTEI = { 207, 31, "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m"
};

static emlrtRTEInfo cb_emlrtRTEI = { 230, 31, "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m"
};

static emlrtRTEInfo ib_emlrtRTEI = { 209, 13, "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m"
};

static emlrtRTEInfo jb_emlrtRTEI = { 210, 13, "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m"
};

static emlrtBCInfo j_emlrtBCI = { -1, -1, 189, 39, "indexOfParamMtxSzTab_",
  "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m",
  0 };

static emlrtBCInfo k_emlrtBCI = { -1, -1, 191, 51, "paramMtxSzTab_",
  "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m",
  0 };

static emlrtBCInfo l_emlrtBCI = { -1, -1, 190, 43, "paramMtxSzTab_",
  "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m",
  0 };

static emlrtBCInfo m_emlrtBCI = { -1, -1, 176, 36, "paramMtxSzTab_",
  "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m",
  0 };

static emlrtBCInfo n_emlrtBCI = { -1, -1, 173, 28, "paramMtxSzTab_",
  "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m",
  0 };

static emlrtBCInfo o_emlrtBCI = { -1, -1, 172, 28, "paramMtxSzTab_",
  "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m",
  0 };

static emlrtRTEInfo vb_emlrtRTEI = { 175, 17, "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m"
};

static emlrtRTEInfo xb_emlrtRTEI = { 44, 19, "assertValidSizeArg",
  "C:\\Program Files\\MATLAB\\R2016a\\toolbox\\eml\\eml\\+coder\\+internal\\assertValidSizeArg.m"
};

static emlrtRTEInfo yb_emlrtRTEI = { 71, 15, "reshape",
  "C:\\Program Files\\MATLAB\\R2016a\\toolbox\\eml\\lib\\matlab\\elmat\\reshape.m"
};

static emlrtBCInfo p_emlrtBCI = { -1, -1, 203, 17, "obj.paramMtxCoefs",
  "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m",
  0 };

static emlrtDCInfo emlrtDCI = { 203, 17, "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m",
  1 };

static emlrtBCInfo q_emlrtBCI = { -1, -1, 199, 50, "obj.indexOfParamMtxSzTab",
  "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m",
  0 };

static emlrtBCInfo r_emlrtBCI = { -1, -1, 198, 50, "obj.indexOfParamMtxSzTab",
  "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m",
  0 };

static emlrtDCInfo b_emlrtDCI = { 68, 62, "reshape",
  "C:\\Program Files\\MATLAB\\R2016a\\toolbox\\eml\\lib\\matlab\\elmat\\reshape.m",
  4 };

static emlrtECInfo h_emlrtECI = { 2, 212, 40, "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m"
};

static emlrtECInfo i_emlrtECI = { -1, 212, 13, "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m"
};

static emlrtECInfo j_emlrtECI = { 2, 213, 40, "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m"
};

static emlrtECInfo k_emlrtECI = { -1, 213, 13, "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m"
};

static emlrtECInfo n_emlrtECI = { -1, 234, 13, "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m"
};

static emlrtBCInfo fb_emlrtBCI = { -1, -1, 234, 35, "arrayCoefs",
  "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m",
  0 };

static emlrtBCInfo gb_emlrtBCI = { -1, -1, 235, 41, "arrayCoefs",
  "AbstNsoltCoefManipulator2d",
  "C:\\Users\\Shogo\\Documents\\MATLAB\\SaivDr\\+saivdr\\+dictionary\\+nsoltx\\AbstNsoltCoefManipulator2d.m",
  0 };

/* Function Definitions */
void c_AbstNsoltCoefManipulator2d_bl(const emlrtStack *sp, emxArray_real_T
  *arrayCoefs)
{
  emxArray_real_T *upper;
  int32_T loop_ub;
  int32_T i47;
  emxArray_real_T *lower;
  int32_T i48;
  emxArray_real_T *b_arrayCoefs;
  int32_T c_arrayCoefs[2];
  emxArray_real_T *d_arrayCoefs;
  int32_T e_arrayCoefs[2];
  emxArray_int32_T *r5;
  emxArray_real_T *f_arrayCoefs;
  int32_T iv26[2];
  int32_T g_arrayCoefs[2];
  int32_T b_upper[2];
  int32_T b_lower[2];
  int32_T iv27[2];
  emlrtHeapReferenceStackEnterFcnR2012b(sp);
  emxInit_real_T1(sp, &upper, 2, &ib_emlrtRTEI, true);
  loop_ub = arrayCoefs->size[1];
  i47 = upper->size[0] * upper->size[1];
  upper->size[0] = 12;
  upper->size[1] = loop_ub;
  emxEnsureCapacity(sp, (emxArray__common *)upper, i47, (int32_T)sizeof(real_T),
                    &ab_emlrtRTEI);
  for (i47 = 0; i47 < loop_ub; i47++) {
    for (i48 = 0; i48 < 12; i48++) {
      upper->data[i48 + upper->size[0] * i47] = arrayCoefs->data[i48 +
        arrayCoefs->size[0] * i47];
    }
  }

  emxInit_real_T1(sp, &lower, 2, &jb_emlrtRTEI, true);
  loop_ub = arrayCoefs->size[1];
  i47 = lower->size[0] * lower->size[1];
  lower->size[0] = 12;
  lower->size[1] = loop_ub;
  emxEnsureCapacity(sp, (emxArray__common *)lower, i47, (int32_T)sizeof(real_T),
                    &ab_emlrtRTEI);
  for (i47 = 0; i47 < loop_ub; i47++) {
    for (i48 = 0; i48 < 12; i48++) {
      lower->data[i48 + lower->size[0] * i47] = arrayCoefs->data[(i48 +
        arrayCoefs->size[0] * i47) + 12];
    }
  }

  emxInit_real_T1(sp, &b_arrayCoefs, 2, &ab_emlrtRTEI, true);
  loop_ub = arrayCoefs->size[1];
  i47 = b_arrayCoefs->size[0] * b_arrayCoefs->size[1];
  b_arrayCoefs->size[0] = 12;
  b_arrayCoefs->size[1] = loop_ub;
  emxEnsureCapacity(sp, (emxArray__common *)b_arrayCoefs, i47, (int32_T)sizeof
                    (real_T), &ab_emlrtRTEI);
  for (i47 = 0; i47 < loop_ub; i47++) {
    for (i48 = 0; i48 < 12; i48++) {
      b_arrayCoefs->data[i48 + b_arrayCoefs->size[0] * i47] = arrayCoefs->
        data[i48 + arrayCoefs->size[0] * i47];
    }
  }

  for (i47 = 0; i47 < 2; i47++) {
    c_arrayCoefs[i47] = b_arrayCoefs->size[i47];
  }

  emxFree_real_T(&b_arrayCoefs);
  emxInit_real_T1(sp, &d_arrayCoefs, 2, &ab_emlrtRTEI, true);
  loop_ub = arrayCoefs->size[1];
  i47 = d_arrayCoefs->size[0] * d_arrayCoefs->size[1];
  d_arrayCoefs->size[0] = 12;
  d_arrayCoefs->size[1] = loop_ub;
  emxEnsureCapacity(sp, (emxArray__common *)d_arrayCoefs, i47, (int32_T)sizeof
                    (real_T), &ab_emlrtRTEI);
  for (i47 = 0; i47 < loop_ub; i47++) {
    for (i48 = 0; i48 < 12; i48++) {
      d_arrayCoefs->data[i48 + d_arrayCoefs->size[0] * i47] = arrayCoefs->data
        [(i48 + arrayCoefs->size[0] * i47) + 12];
    }
  }

  for (i47 = 0; i47 < 2; i47++) {
    e_arrayCoefs[i47] = d_arrayCoefs->size[i47];
  }

  emxFree_real_T(&d_arrayCoefs);
  emxInit_int32_T1(sp, &r5, 1, &ab_emlrtRTEI, true);
  if ((c_arrayCoefs[0] != e_arrayCoefs[0]) || (c_arrayCoefs[1] != e_arrayCoefs[1]))
  {
    emlrtSizeEqCheckNDR2012b(&c_arrayCoefs[0], &e_arrayCoefs[0], &h_emlrtECI, sp);
  }

  loop_ub = arrayCoefs->size[1];
  i47 = r5->size[0];
  r5->size[0] = loop_ub;
  emxEnsureCapacity(sp, (emxArray__common *)r5, i47, (int32_T)sizeof(int32_T),
                    &ab_emlrtRTEI);
  for (i47 = 0; i47 < loop_ub; i47++) {
    r5->data[i47] = i47;
  }

  emxInit_real_T1(sp, &f_arrayCoefs, 2, &ab_emlrtRTEI, true);
  iv26[0] = 12;
  iv26[1] = r5->size[0];
  loop_ub = arrayCoefs->size[1];
  i47 = f_arrayCoefs->size[0] * f_arrayCoefs->size[1];
  f_arrayCoefs->size[0] = 12;
  f_arrayCoefs->size[1] = loop_ub;
  emxEnsureCapacity(sp, (emxArray__common *)f_arrayCoefs, i47, (int32_T)sizeof
                    (real_T), &ab_emlrtRTEI);
  for (i47 = 0; i47 < loop_ub; i47++) {
    for (i48 = 0; i48 < 12; i48++) {
      f_arrayCoefs->data[i48 + f_arrayCoefs->size[0] * i47] = arrayCoefs->
        data[i48 + arrayCoefs->size[0] * i47];
    }
  }

  for (i47 = 0; i47 < 2; i47++) {
    g_arrayCoefs[i47] = f_arrayCoefs->size[i47];
  }

  emxFree_real_T(&f_arrayCoefs);
  emlrtSubAssignSizeCheckR2012b(iv26, 2, g_arrayCoefs, 2, &i_emlrtECI, sp);
  loop_ub = upper->size[1];
  for (i47 = 0; i47 < loop_ub; i47++) {
    for (i48 = 0; i48 < 12; i48++) {
      arrayCoefs->data[i48 + arrayCoefs->size[0] * r5->data[i47]] = upper->
        data[i48 + upper->size[0] * i47] + lower->data[i48 + lower->size[0] *
        i47];
    }
  }

  for (i47 = 0; i47 < 2; i47++) {
    b_upper[i47] = upper->size[i47];
  }

  for (i47 = 0; i47 < 2; i47++) {
    b_lower[i47] = lower->size[i47];
  }

  if ((b_upper[0] != b_lower[0]) || (b_upper[1] != b_lower[1])) {
    emlrtSizeEqCheckNDR2012b(&b_upper[0], &b_lower[0], &j_emlrtECI, sp);
  }

  loop_ub = arrayCoefs->size[1];
  i47 = r5->size[0];
  r5->size[0] = loop_ub;
  emxEnsureCapacity(sp, (emxArray__common *)r5, i47, (int32_T)sizeof(int32_T),
                    &ab_emlrtRTEI);
  for (i47 = 0; i47 < loop_ub; i47++) {
    r5->data[i47] = i47;
  }

  iv27[0] = 12;
  iv27[1] = r5->size[0];
  emlrtSubAssignSizeCheckR2012b(iv27, 2, *(int32_T (*)[2])upper->size, 2,
    &k_emlrtECI, sp);
  loop_ub = upper->size[1];
  for (i47 = 0; i47 < loop_ub; i47++) {
    for (i48 = 0; i48 < 12; i48++) {
      arrayCoefs->data[(i48 + arrayCoefs->size[0] * r5->data[i47]) + 12] =
        upper->data[i48 + upper->size[0] * i47] - lower->data[i48 + lower->size
        [0] * i47];
    }
  }

  emxFree_int32_T(&r5);
  emxFree_real_T(&lower);
  emxFree_real_T(&upper);
  emlrtHeapReferenceStackLeaveFcnR2012b(sp);
}

void c_AbstNsoltCoefManipulator2d_ge(const emlrtStack *sp, const
  c_saivdr_dictionary_nsoltx_desi *obj, emxArray_real_T *value)
{
  int32_T i8;
  real_T startIdx;
  real_T dimension[2];
  real_T endIdx;
  int32_T maxdimlen;
  int32_T i9;
  emxArray_real_T *x;
  int32_T sz[2];
  int32_T exitg2;
  boolean_T overflow;
  boolean_T guard1 = false;
  int32_T exitg1;
  real_T n;
  emlrtStack st;
  emlrtStack b_st;
  emlrtStack c_st;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b(sp);
  i8 = obj->indexOfParamMtxSzTab->size[0];
  if (!(1 <= i8)) {
    emlrtDynamicBoundsCheckR2012b(1, 1, i8, &r_emlrtBCI, sp);
  }

  startIdx = obj->indexOfParamMtxSzTab->data[0];
  i8 = obj->indexOfParamMtxSzTab->size[0];
  if (!(1 <= i8)) {
    emlrtDynamicBoundsCheckR2012b(1, 1, i8, &q_emlrtBCI, sp);
  }

  for (i8 = 0; i8 < 2; i8++) {
    dimension[i8] = obj->indexOfParamMtxSzTab->data[obj->
      indexOfParamMtxSzTab->size[0] * (1 + i8)];
  }

  endIdx = (startIdx + dimension[0] * dimension[1]) - 1.0;
  if (startIdx > endIdx) {
    i8 = 0;
    maxdimlen = 0;
  } else {
    i8 = obj->paramMtxCoefs->size[0];
    if (startIdx != (int32_T)muDoubleScalarFloor(startIdx)) {
      emlrtIntegerCheckR2012b(startIdx, &emlrtDCI, sp);
    }

    i9 = (int32_T)startIdx;
    if (!((i9 >= 1) && (i9 <= i8))) {
      emlrtDynamicBoundsCheckR2012b(i9, 1, i8, &p_emlrtBCI, sp);
    }

    i8 = i9 - 1;
    i9 = obj->paramMtxCoefs->size[0];
    if (endIdx != (int32_T)muDoubleScalarFloor(endIdx)) {
      emlrtIntegerCheckR2012b(endIdx, &emlrtDCI, sp);
    }

    maxdimlen = (int32_T)endIdx;
    if (!((maxdimlen >= 1) && (maxdimlen <= i9))) {
      emlrtDynamicBoundsCheckR2012b(maxdimlen, 1, i9, &p_emlrtBCI, sp);
    }
  }

  emxInit_real_T(sp, &x, 1, &s_emlrtRTEI, true);
  sz[0] = 1;
  sz[1] = maxdimlen - i8;
  st.site = &sc_emlrtRSI;
  indexShapeCheck(&st, obj->paramMtxCoefs->size[0], sz);
  st.site = &tc_emlrtRSI;
  i9 = x->size[0];
  x->size[0] = maxdimlen - i8;
  emxEnsureCapacity(&st, (emxArray__common *)x, i9, (int32_T)sizeof(real_T),
                    &s_emlrtRTEI);
  maxdimlen -= i8;
  for (i9 = 0; i9 < maxdimlen; i9++) {
    x->data[i9] = obj->paramMtxCoefs->data[i8 + i9];
  }

  b_st.site = &vc_emlrtRSI;
  maxdimlen = 0;
  do {
    exitg2 = 0;
    if (maxdimlen < 2) {
      if ((dimension[maxdimlen] != muDoubleScalarFloor(dimension[maxdimlen])) ||
          muDoubleScalarIsInf(dimension[maxdimlen])) {
        overflow = false;
        exitg2 = 1;
      } else {
        maxdimlen++;
      }
    } else {
      overflow = true;
      exitg2 = 1;
    }
  } while (exitg2 == 0);

  guard1 = false;
  if (overflow) {
    maxdimlen = 0;
    do {
      exitg1 = 0;
      if (maxdimlen < 2) {
        if ((-2.147483648E+9 > dimension[maxdimlen]) || (2.147483647E+9 <
             dimension[maxdimlen])) {
          overflow = false;
          exitg1 = 1;
        } else {
          maxdimlen++;
        }
      } else {
        overflow = true;
        exitg1 = 1;
      }
    } while (exitg1 == 0);

    if (overflow) {
      overflow = true;
    } else {
      guard1 = true;
    }
  } else {
    guard1 = true;
  }

  if (guard1) {
    overflow = false;
  }

  if (overflow) {
  } else {
    emlrtErrorWithMessageIdR2012b(&b_st, &xb_emlrtRTEI,
      "Coder:toolbox:eml_assert_valid_size_arg_invalidSizeVector", 4, 12,
      MIN_int32_T, 12, MAX_int32_T);
  }

  n = 1.0;
  for (maxdimlen = 0; maxdimlen < 2; maxdimlen++) {
    if (dimension[maxdimlen] <= 0.0) {
      n = 0.0;
    } else {
      n *= dimension[maxdimlen];
    }
  }

  if (2.147483647E+9 >= n) {
  } else {
    emlrtErrorWithMessageIdR2012b(&b_st, &wb_emlrtRTEI, "Coder:MATLAB:pmaxsize",
      0);
  }

  for (i8 = 0; i8 < 2; i8++) {
    sz[i8] = (int32_T)dimension[i8];
  }

  b_st.site = &wc_emlrtRSI;
  dimension[0] = x->size[0];
  maxdimlen = (int32_T)(uint32_T)dimension[0];
  if (1 > (int32_T)(uint32_T)dimension[0]) {
    maxdimlen = 1;
  }

  maxdimlen = muIntScalarMax_sint32(x->size[0], maxdimlen);
  if (sz[0] > maxdimlen) {
    b_st.site = &xc_emlrtRSI;
    error(&b_st);
  }

  if (sz[1] > maxdimlen) {
    b_st.site = &xc_emlrtRSI;
    error(&b_st);
  }

  for (i8 = 0; i8 < 2; i8++) {
    i9 = sz[i8];
    if (!(i9 >= 0)) {
      emlrtNonNegativeCheckR2012b(i9, &b_emlrtDCI, &st);
    }
  }

  i8 = value->size[0] * value->size[1];
  value->size[0] = sz[0];
  value->size[1] = sz[1];
  emxEnsureCapacity(&st, (emxArray__common *)value, i8, (int32_T)sizeof(real_T),
                    &s_emlrtRTEI);
  if (x->size[0] == value->size[0] * value->size[1]) {
  } else {
    emlrtErrorWithMessageIdR2012b(&st, &yb_emlrtRTEI,
      "Coder:MATLAB:getReshapeDims_notSameNumel", 0);
  }

  b_st.site = &yc_emlrtRSI;
  overflow = ((!(1 > x->size[0])) && (x->size[0] > 2147483646));
  if (overflow) {
    c_st.site = &cb_emlrtRSI;
    b_check_forloop_overflow_error(&c_st, true);
  }

  for (maxdimlen = 0; maxdimlen + 1 <= x->size[0]; maxdimlen++) {
    value->data[maxdimlen] = x->data[maxdimlen];
  }

  emxFree_real_T(&x);
  emlrtHeapReferenceStackLeaveFcnR2012b(sp);
}

void c_AbstNsoltCoefManipulator2d_lo(const emlrtStack *sp, const
  d_saivdr_dictionary_nsoltx_desi *obj, emxArray_real_T *arrayCoefs, uint32_T
  iCol, const emxArray_real_T *U)
{
  emxArray_int32_T *r7;
  uint32_T nRows_;
  uint32_T q0;
  uint32_T qY;
  uint64_T u7;
  uint32_T indexCol;
  int32_T b_arrayCoefs;
  int32_T i51;
  int32_T loop_ub;
  emxArray_real_T *b;
  int32_T i52;
  boolean_T innerDimOk;
  int32_T i53;
  emxArray_real_T *C;
  int32_T iv30[2];
  real_T alpha1;
  int32_T b_loop_ub;
  real_T beta1;
  char_T TRANSB;
  char_T TRANSA;
  ptrdiff_t m_t;
  ptrdiff_t n_t;
  ptrdiff_t k_t;
  ptrdiff_t lda_t;
  ptrdiff_t ldb_t;
  ptrdiff_t ldc_t;
  emlrtStack st;
  emlrtStack b_st;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b(sp);
  emxInit_int32_T1(sp, &r7, 1, &cb_emlrtRTEI, true);
  nRows_ = obj->nRows + MAX_uint32_T;
  q0 = iCol;
  qY = q0 - 1U;
  if (qY > q0) {
    qY = 0U;
  }

  u7 = (uint64_T)qY * (nRows_ - MAX_uint32_T);
  if (u7 > 4294967295ULL) {
    u7 = 4294967295ULL;
  }

  indexCol = (uint32_T)u7;
  b_arrayCoefs = arrayCoefs->size[1];
  i51 = r7->size[0];
  r7->size[0] = (int32_T)nRows_ + 1;
  emxEnsureCapacity(sp, (emxArray__common *)r7, i51, (int32_T)sizeof(int32_T),
                    &cb_emlrtRTEI);
  loop_ub = (int32_T)nRows_;
  for (i51 = 0; i51 <= loop_ub; i51++) {
    u7 = (uint64_T)indexCol + (1U + i51);
    if (u7 > 4294967295ULL) {
      u7 = 4294967295ULL;
    }

    i52 = (int32_T)u7;
    if (!((i52 >= 1) && (i52 <= b_arrayCoefs))) {
      emlrtDynamicBoundsCheckR2012b(i52, 1, b_arrayCoefs, &fb_emlrtBCI, sp);
    }

    r7->data[i51] = i52 - 1;
  }

  emxInit_real_T1(sp, &b, 2, &cb_emlrtRTEI, true);
  st.site = &be_emlrtRSI;
  b_arrayCoefs = arrayCoefs->size[1];
  i51 = b->size[0] * b->size[1];
  b->size[0] = 12;
  b->size[1] = (int32_T)nRows_ + 1;
  emxEnsureCapacity(&st, (emxArray__common *)b, i51, (int32_T)sizeof(real_T),
                    &cb_emlrtRTEI);
  loop_ub = (int32_T)nRows_;
  for (i51 = 0; i51 <= loop_ub; i51++) {
    for (i52 = 0; i52 < 12; i52++) {
      i53 = 1 + i51;
      if (i53 < 0) {
        i53 = 0;
      }

      u7 = (uint64_T)indexCol + i53;
      if (u7 > 4294967295ULL) {
        u7 = 4294967295ULL;
      }

      i53 = (int32_T)u7;
      if (!((i53 >= 1) && (i53 <= b_arrayCoefs))) {
        emlrtDynamicBoundsCheckR2012b(i53, 1, b_arrayCoefs, &gb_emlrtBCI, &st);
      }

      b->data[i52 + b->size[0] * i51] = arrayCoefs->data[(i52 + arrayCoefs->
        size[0] * (i53 - 1)) + 12];
    }
  }

  b_st.site = &ed_emlrtRSI;
  innerDimOk = (U->size[1] == 12);
  if (!innerDimOk) {
    if ((U->size[0] == 1) && (U->size[1] == 1)) {
      emlrtErrorWithMessageIdR2012b(&b_st, &sb_emlrtRTEI,
        "Coder:toolbox:mtimes_noDynamicScalarExpansion", 0);
    } else {
      emlrtErrorWithMessageIdR2012b(&b_st, &tb_emlrtRTEI,
        "Coder:MATLAB:innerdim", 0);
    }
  }

  emxInit_real_T1(&st, &C, 2, &cb_emlrtRTEI, true);
  if (U->size[1] == 1) {
    i51 = C->size[0] * C->size[1];
    C->size[0] = U->size[0];
    C->size[1] = b->size[1];
    emxEnsureCapacity(&st, (emxArray__common *)C, i51, (int32_T)sizeof(real_T),
                      &cb_emlrtRTEI);
    loop_ub = U->size[0];
    for (i51 = 0; i51 < loop_ub; i51++) {
      b_arrayCoefs = b->size[1];
      for (i52 = 0; i52 < b_arrayCoefs; i52++) {
        C->data[i51 + C->size[0] * i52] = 0.0;
        b_loop_ub = U->size[1];
        for (i53 = 0; i53 < b_loop_ub; i53++) {
          C->data[i51 + C->size[0] * i52] += U->data[i51 + U->size[0] * i53] *
            b->data[i53 + b->size[0] * i52];
        }
      }
    }
  } else {
    q0 = (uint32_T)U->size[0];
    i51 = C->size[0] * C->size[1];
    C->size[0] = (int32_T)q0;
    C->size[1] = (int32_T)nRows_ + 1;
    emxEnsureCapacity(&st, (emxArray__common *)C, i51, (int32_T)sizeof(real_T),
                      &cb_emlrtRTEI);
    loop_ub = C->size[1];
    for (i51 = 0; i51 < loop_ub; i51++) {
      b_arrayCoefs = C->size[0];
      for (i52 = 0; i52 < b_arrayCoefs; i52++) {
        C->data[i52 + C->size[0] * i51] = 0.0;
      }
    }

    b_st.site = &y_emlrtRSI;
    if ((U->size[0] < 1) || ((int32_T)nRows_ + 1 < 1) || (U->size[1] < 1)) {
    } else {
      alpha1 = 1.0;
      beta1 = 0.0;
      TRANSB = 'N';
      TRANSA = 'N';
      m_t = (ptrdiff_t)U->size[0];
      n_t = (ptrdiff_t)((int32_T)nRows_ + 1);
      k_t = (ptrdiff_t)U->size[1];
      lda_t = (ptrdiff_t)U->size[0];
      ldb_t = (ptrdiff_t)U->size[1];
      ldc_t = (ptrdiff_t)U->size[0];
      dgemm(&TRANSA, &TRANSB, &m_t, &n_t, &k_t, &alpha1, &U->data[0], &lda_t,
            &b->data[0], &ldb_t, &beta1, &C->data[0], &ldc_t);
    }
  }

  emxFree_real_T(&b);
  iv30[0] = 12;
  iv30[1] = r7->size[0];
  emlrtSubAssignSizeCheckR2012b(iv30, 2, *(int32_T (*)[2])C->size, 2,
    &n_emlrtECI, sp);
  b_arrayCoefs = r7->size[0];
  for (i51 = 0; i51 < b_arrayCoefs; i51++) {
    for (i52 = 0; i52 < 12; i52++) {
      arrayCoefs->data[(i52 + arrayCoefs->size[0] * r7->data[i51]) + 12] =
        C->data[i52 + 12 * i51];
    }
  }

  emxFree_real_T(&C);
  emxFree_int32_T(&r7);
  emlrtHeapReferenceStackLeaveFcnR2012b(sp);
}

void c_AbstNsoltCoefManipulator2d_se(const emlrtStack *sp,
  c_saivdr_dictionary_nsoltx_desi *obj)
{
  uint32_T ord[2];
  int32_T i6;
  emxArray_real_T *paramMtxSzTab_;
  real_T y;
  int32_T loop_ub;
  int32_T iOrd;
  emxArray_real_T *indexOfParamMtxSzTab_;
  real_T cidx;
  uint32_T b;
  uint32_T iRow;
  int32_T b_indexOfParamMtxSzTab_;
  emlrtHeapReferenceStackEnterFcnR2012b(sp);
  for (i6 = 0; i6 < 2; i6++) {
    ord[i6] = obj->PolyPhaseOrder[i6];
  }

  emxInit_real_T1(sp, &paramMtxSzTab_, 2, &o_emlrtRTEI, true);
  y = (real_T)ord[0] + (real_T)ord[1];
  i6 = paramMtxSzTab_->size[0] * paramMtxSzTab_->size[1];
  paramMtxSzTab_->size[0] = (int32_T)(y + 2.0);
  paramMtxSzTab_->size[1] = 2;
  emxEnsureCapacity(sp, (emxArray__common *)paramMtxSzTab_, i6, (int32_T)sizeof
                    (real_T), &n_emlrtRTEI);
  loop_ub = (int32_T)(y + 2.0) << 1;
  for (i6 = 0; i6 < loop_ub; i6++) {
    paramMtxSzTab_->data[i6] = 0.0;
  }

  i6 = (int32_T)(y + 2.0);
  if (!(1 <= i6)) {
    emlrtDynamicBoundsCheckR2012b(1, 1, i6, &o_emlrtBCI, sp);
  }

  for (i6 = 0; i6 < 2; i6++) {
    paramMtxSzTab_->data[paramMtxSzTab_->size[0] * i6] = 12.0;
  }

  i6 = paramMtxSzTab_->size[0];
  if (!(2 <= i6)) {
    emlrtDynamicBoundsCheckR2012b(2, 1, i6, &n_emlrtBCI, sp);
  }

  for (i6 = 0; i6 < 2; i6++) {
    paramMtxSzTab_->data[1 + paramMtxSzTab_->size[0] * i6] = 12.0;
  }

  y = (real_T)ord[0] + (real_T)ord[1];
  emlrtForLoopVectorCheckR2012b(1.0, 1.0, y, mxDOUBLE_CLASS, (int32_T)y,
    &vb_emlrtRTEI, sp);
  iOrd = 0;
  while (iOrd <= (int32_T)y - 1) {
    loop_ub = paramMtxSzTab_->size[0];
    i6 = (int32_T)((1.0 + (real_T)iOrd) + 2.0);
    if (!((i6 >= 1) && (i6 <= loop_ub))) {
      emlrtDynamicBoundsCheckR2012b(i6, 1, loop_ub, &m_emlrtBCI, sp);
    }

    for (i6 = 0; i6 < 2; i6++) {
      paramMtxSzTab_->data[((int32_T)((1.0 + (real_T)iOrd) + 2.0) +
                            paramMtxSzTab_->size[0] * i6) - 1] = 12.0;
    }

    iOrd++;
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b(sp);
    }
  }

  emxInit_real_T1(sp, &indexOfParamMtxSzTab_, 2, &p_emlrtRTEI, true);
  i6 = indexOfParamMtxSzTab_->size[0] * indexOfParamMtxSzTab_->size[1];
  indexOfParamMtxSzTab_->size[0] = paramMtxSzTab_->size[0];
  indexOfParamMtxSzTab_->size[1] = 3;
  emxEnsureCapacity(sp, (emxArray__common *)indexOfParamMtxSzTab_, i6, (int32_T)
                    sizeof(real_T), &n_emlrtRTEI);
  loop_ub = paramMtxSzTab_->size[0] * 3;
  for (i6 = 0; i6 < loop_ub; i6++) {
    indexOfParamMtxSzTab_->data[i6] = 0.0;
  }

  cidx = 1.0;
  b = (uint32_T)paramMtxSzTab_->size[0];
  iRow = 1U;
  while (iRow <= b) {
    i6 = paramMtxSzTab_->size[0];
    loop_ub = (int32_T)iRow;
    if (!((loop_ub >= 1) && (loop_ub <= i6))) {
      emlrtDynamicBoundsCheckR2012b(loop_ub, 1, i6, &l_emlrtBCI, sp);
    }

    b_indexOfParamMtxSzTab_ = indexOfParamMtxSzTab_->size[0];
    i6 = (int32_T)iRow;
    if (!((i6 >= 1) && (i6 <= b_indexOfParamMtxSzTab_))) {
      emlrtDynamicBoundsCheckR2012b(i6, 1, b_indexOfParamMtxSzTab_, &j_emlrtBCI,
        sp);
    }

    indexOfParamMtxSzTab_->data[(int32_T)iRow - 1] = cidx;
    for (i6 = 0; i6 < 2; i6++) {
      indexOfParamMtxSzTab_->data[((int32_T)iRow + indexOfParamMtxSzTab_->size[0]
        * (i6 + 1)) - 1] = paramMtxSzTab_->data[(loop_ub + paramMtxSzTab_->size
        [0] * i6) - 1];
    }

    i6 = paramMtxSzTab_->size[0];
    loop_ub = (int32_T)iRow;
    if (!((loop_ub >= 1) && (loop_ub <= i6))) {
      emlrtDynamicBoundsCheckR2012b(loop_ub, 1, i6, &k_emlrtBCI, sp);
    }

    cidx += paramMtxSzTab_->data[(int32_T)iRow - 1] * paramMtxSzTab_->data
      [((int32_T)iRow + paramMtxSzTab_->size[0]) - 1];
    iRow++;
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b(sp);
    }
  }

  emxFree_real_T(&paramMtxSzTab_);
  i6 = obj->indexOfParamMtxSzTab->size[0] * obj->indexOfParamMtxSzTab->size[1];
  obj->indexOfParamMtxSzTab->size[0] = indexOfParamMtxSzTab_->size[0];
  obj->indexOfParamMtxSzTab->size[1] = 3;
  emxEnsureCapacity(sp, (emxArray__common *)obj->indexOfParamMtxSzTab, i6,
                    (int32_T)sizeof(real_T), &n_emlrtRTEI);
  loop_ub = indexOfParamMtxSzTab_->size[0] * indexOfParamMtxSzTab_->size[1];
  for (i6 = 0; i6 < loop_ub; i6++) {
    obj->indexOfParamMtxSzTab->data[i6] = indexOfParamMtxSzTab_->data[i6];
  }

  emxFree_real_T(&indexOfParamMtxSzTab_);
  emlrtHeapReferenceStackLeaveFcnR2012b(sp);
}

void c_AbstNsoltCoefManipulator2d_st(const emlrtStack *sp,
  c_saivdr_dictionary_nsoltx_desi *obj, const uint32_T subScale[2], const
  emxArray_real_T *pmCoefs)
{
  int32_T i7;
  int32_T loop_ub;
  emlrtStack st;
  st.prev = sp;
  st.tls = sp->tls;
  i7 = obj->paramMtxCoefs->size[0];
  obj->paramMtxCoefs->size[0] = pmCoefs->size[0];
  emxEnsureCapacity(sp, (emxArray__common *)obj->paramMtxCoefs, i7, (int32_T)
                    sizeof(real_T), &q_emlrtRTEI);
  loop_ub = pmCoefs->size[0];
  for (i7 = 0; i7 < loop_ub; i7++) {
    obj->paramMtxCoefs->data[i7] = pmCoefs->data[i7];
  }

  st.site = &nc_emlrtRSI;
  if (subScale[0] > 0U) {
  } else {
    emlrtErrorWithMessageIdR2012b(&st, &ub_emlrtRTEI,
      "MATLAB:system:positiveIntMustBePosIntValuedScalar", 3, 4, 5, "nRows");
  }

  obj->nRows = subScale[0];
  st.site = &oc_emlrtRSI;
  if (subScale[1] > 0U) {
  } else {
    emlrtErrorWithMessageIdR2012b(&st, &ub_emlrtRTEI,
      "MATLAB:system:positiveIntMustBePosIntValuedScalar", 3, 4, 5, "nCols");
  }

  obj->nCols = subScale[1];
}

void d_AbstNsoltCoefManipulator2d_bl(const emlrtStack *sp, emxArray_real_T
  *arrayCoefs)
{
  emxArray_real_T *upper;
  int32_T loop_ub;
  int32_T i70;
  emxArray_real_T *lower;
  int32_T i71;
  emxArray_real_T *b_arrayCoefs;
  int32_T c_arrayCoefs[2];
  emxArray_real_T *d_arrayCoefs;
  int32_T e_arrayCoefs[2];
  emxArray_int32_T *r12;
  emxArray_real_T *f_arrayCoefs;
  int32_T iv40[2];
  int32_T g_arrayCoefs[2];
  int32_T b_upper[2];
  int32_T b_lower[2];
  int32_T iv41[2];
  emlrtHeapReferenceStackEnterFcnR2012b(sp);
  emxInit_real_T1(sp, &upper, 2, &ib_emlrtRTEI, true);
  loop_ub = arrayCoefs->size[1];
  i70 = upper->size[0] * upper->size[1];
  upper->size[0] = 12;
  upper->size[1] = loop_ub;
  emxEnsureCapacity(sp, (emxArray__common *)upper, i70, (int32_T)sizeof(real_T),
                    &ab_emlrtRTEI);
  for (i70 = 0; i70 < loop_ub; i70++) {
    for (i71 = 0; i71 < 12; i71++) {
      upper->data[i71 + upper->size[0] * i70] = arrayCoefs->data[i71 +
        arrayCoefs->size[0] * i70];
    }
  }

  emxInit_real_T1(sp, &lower, 2, &jb_emlrtRTEI, true);
  loop_ub = arrayCoefs->size[1];
  i70 = lower->size[0] * lower->size[1];
  lower->size[0] = 12;
  lower->size[1] = loop_ub;
  emxEnsureCapacity(sp, (emxArray__common *)lower, i70, (int32_T)sizeof(real_T),
                    &ab_emlrtRTEI);
  for (i70 = 0; i70 < loop_ub; i70++) {
    for (i71 = 0; i71 < 12; i71++) {
      lower->data[i71 + lower->size[0] * i70] = arrayCoefs->data[(i71 +
        arrayCoefs->size[0] * i70) + 12];
    }
  }

  emxInit_real_T1(sp, &b_arrayCoefs, 2, &ab_emlrtRTEI, true);
  loop_ub = arrayCoefs->size[1];
  i70 = b_arrayCoefs->size[0] * b_arrayCoefs->size[1];
  b_arrayCoefs->size[0] = 12;
  b_arrayCoefs->size[1] = loop_ub;
  emxEnsureCapacity(sp, (emxArray__common *)b_arrayCoefs, i70, (int32_T)sizeof
                    (real_T), &ab_emlrtRTEI);
  for (i70 = 0; i70 < loop_ub; i70++) {
    for (i71 = 0; i71 < 12; i71++) {
      b_arrayCoefs->data[i71 + b_arrayCoefs->size[0] * i70] = arrayCoefs->
        data[i71 + arrayCoefs->size[0] * i70];
    }
  }

  for (i70 = 0; i70 < 2; i70++) {
    c_arrayCoefs[i70] = b_arrayCoefs->size[i70];
  }

  emxFree_real_T(&b_arrayCoefs);
  emxInit_real_T1(sp, &d_arrayCoefs, 2, &ab_emlrtRTEI, true);
  loop_ub = arrayCoefs->size[1];
  i70 = d_arrayCoefs->size[0] * d_arrayCoefs->size[1];
  d_arrayCoefs->size[0] = 12;
  d_arrayCoefs->size[1] = loop_ub;
  emxEnsureCapacity(sp, (emxArray__common *)d_arrayCoefs, i70, (int32_T)sizeof
                    (real_T), &ab_emlrtRTEI);
  for (i70 = 0; i70 < loop_ub; i70++) {
    for (i71 = 0; i71 < 12; i71++) {
      d_arrayCoefs->data[i71 + d_arrayCoefs->size[0] * i70] = arrayCoefs->data
        [(i71 + arrayCoefs->size[0] * i70) + 12];
    }
  }

  for (i70 = 0; i70 < 2; i70++) {
    e_arrayCoefs[i70] = d_arrayCoefs->size[i70];
  }

  emxFree_real_T(&d_arrayCoefs);
  emxInit_int32_T1(sp, &r12, 1, &ab_emlrtRTEI, true);
  if ((c_arrayCoefs[0] != e_arrayCoefs[0]) || (c_arrayCoefs[1] != e_arrayCoefs[1]))
  {
    emlrtSizeEqCheckNDR2012b(&c_arrayCoefs[0], &e_arrayCoefs[0], &h_emlrtECI, sp);
  }

  loop_ub = arrayCoefs->size[1];
  i70 = r12->size[0];
  r12->size[0] = loop_ub;
  emxEnsureCapacity(sp, (emxArray__common *)r12, i70, (int32_T)sizeof(int32_T),
                    &ab_emlrtRTEI);
  for (i70 = 0; i70 < loop_ub; i70++) {
    r12->data[i70] = i70;
  }

  emxInit_real_T1(sp, &f_arrayCoefs, 2, &ab_emlrtRTEI, true);
  iv40[0] = 12;
  iv40[1] = r12->size[0];
  loop_ub = arrayCoefs->size[1];
  i70 = f_arrayCoefs->size[0] * f_arrayCoefs->size[1];
  f_arrayCoefs->size[0] = 12;
  f_arrayCoefs->size[1] = loop_ub;
  emxEnsureCapacity(sp, (emxArray__common *)f_arrayCoefs, i70, (int32_T)sizeof
                    (real_T), &ab_emlrtRTEI);
  for (i70 = 0; i70 < loop_ub; i70++) {
    for (i71 = 0; i71 < 12; i71++) {
      f_arrayCoefs->data[i71 + f_arrayCoefs->size[0] * i70] = arrayCoefs->
        data[i71 + arrayCoefs->size[0] * i70];
    }
  }

  for (i70 = 0; i70 < 2; i70++) {
    g_arrayCoefs[i70] = f_arrayCoefs->size[i70];
  }

  emxFree_real_T(&f_arrayCoefs);
  emlrtSubAssignSizeCheckR2012b(iv40, 2, g_arrayCoefs, 2, &i_emlrtECI, sp);
  loop_ub = upper->size[1];
  for (i70 = 0; i70 < loop_ub; i70++) {
    for (i71 = 0; i71 < 12; i71++) {
      arrayCoefs->data[i71 + arrayCoefs->size[0] * r12->data[i70]] = upper->
        data[i71 + upper->size[0] * i70] + lower->data[i71 + lower->size[0] *
        i70];
    }
  }

  for (i70 = 0; i70 < 2; i70++) {
    b_upper[i70] = upper->size[i70];
  }

  for (i70 = 0; i70 < 2; i70++) {
    b_lower[i70] = lower->size[i70];
  }

  if ((b_upper[0] != b_lower[0]) || (b_upper[1] != b_lower[1])) {
    emlrtSizeEqCheckNDR2012b(&b_upper[0], &b_lower[0], &j_emlrtECI, sp);
  }

  loop_ub = arrayCoefs->size[1];
  i70 = r12->size[0];
  r12->size[0] = loop_ub;
  emxEnsureCapacity(sp, (emxArray__common *)r12, i70, (int32_T)sizeof(int32_T),
                    &ab_emlrtRTEI);
  for (i70 = 0; i70 < loop_ub; i70++) {
    r12->data[i70] = i70;
  }

  iv41[0] = 12;
  iv41[1] = r12->size[0];
  emlrtSubAssignSizeCheckR2012b(iv41, 2, *(int32_T (*)[2])upper->size, 2,
    &k_emlrtECI, sp);
  loop_ub = upper->size[1];
  for (i70 = 0; i70 < loop_ub; i70++) {
    for (i71 = 0; i71 < 12; i71++) {
      arrayCoefs->data[(i71 + arrayCoefs->size[0] * r12->data[i70]) + 12] =
        upper->data[i71 + upper->size[0] * i70] - lower->data[i71 + lower->size
        [0] * i70];
    }
  }

  emxFree_int32_T(&r12);
  emxFree_real_T(&lower);
  emxFree_real_T(&upper);
  emlrtHeapReferenceStackLeaveFcnR2012b(sp);
}

void d_AbstNsoltCoefManipulator2d_ge(const emlrtStack *sp, const
  c_saivdr_dictionary_nsoltx_desi *obj, emxArray_real_T *value)
{
  int32_T i10;
  real_T startIdx;
  real_T dimension[2];
  real_T endIdx;
  int32_T maxdimlen;
  int32_T i11;
  emxArray_real_T *x;
  int32_T sz[2];
  int32_T exitg2;
  boolean_T overflow;
  boolean_T guard1 = false;
  int32_T exitg1;
  real_T n;
  emlrtStack st;
  emlrtStack b_st;
  emlrtStack c_st;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b(sp);
  i10 = obj->indexOfParamMtxSzTab->size[0];
  if (!(2 <= i10)) {
    emlrtDynamicBoundsCheckR2012b(2, 1, i10, &r_emlrtBCI, sp);
  }

  startIdx = obj->indexOfParamMtxSzTab->data[1];
  i10 = obj->indexOfParamMtxSzTab->size[0];
  if (!(2 <= i10)) {
    emlrtDynamicBoundsCheckR2012b(2, 1, i10, &q_emlrtBCI, sp);
  }

  for (i10 = 0; i10 < 2; i10++) {
    dimension[i10] = obj->indexOfParamMtxSzTab->data[1 +
      obj->indexOfParamMtxSzTab->size[0] * (1 + i10)];
  }

  endIdx = (startIdx + dimension[0] * dimension[1]) - 1.0;
  if (startIdx > endIdx) {
    i10 = 0;
    maxdimlen = 0;
  } else {
    i10 = obj->paramMtxCoefs->size[0];
    if (startIdx != (int32_T)muDoubleScalarFloor(startIdx)) {
      emlrtIntegerCheckR2012b(startIdx, &emlrtDCI, sp);
    }

    i11 = (int32_T)startIdx;
    if (!((i11 >= 1) && (i11 <= i10))) {
      emlrtDynamicBoundsCheckR2012b(i11, 1, i10, &p_emlrtBCI, sp);
    }

    i10 = i11 - 1;
    i11 = obj->paramMtxCoefs->size[0];
    if (endIdx != (int32_T)muDoubleScalarFloor(endIdx)) {
      emlrtIntegerCheckR2012b(endIdx, &emlrtDCI, sp);
    }

    maxdimlen = (int32_T)endIdx;
    if (!((maxdimlen >= 1) && (maxdimlen <= i11))) {
      emlrtDynamicBoundsCheckR2012b(maxdimlen, 1, i11, &p_emlrtBCI, sp);
    }
  }

  emxInit_real_T(sp, &x, 1, &s_emlrtRTEI, true);
  sz[0] = 1;
  sz[1] = maxdimlen - i10;
  st.site = &sc_emlrtRSI;
  indexShapeCheck(&st, obj->paramMtxCoefs->size[0], sz);
  st.site = &tc_emlrtRSI;
  i11 = x->size[0];
  x->size[0] = maxdimlen - i10;
  emxEnsureCapacity(&st, (emxArray__common *)x, i11, (int32_T)sizeof(real_T),
                    &s_emlrtRTEI);
  maxdimlen -= i10;
  for (i11 = 0; i11 < maxdimlen; i11++) {
    x->data[i11] = obj->paramMtxCoefs->data[i10 + i11];
  }

  b_st.site = &vc_emlrtRSI;
  maxdimlen = 0;
  do {
    exitg2 = 0;
    if (maxdimlen < 2) {
      if ((dimension[maxdimlen] != muDoubleScalarFloor(dimension[maxdimlen])) ||
          muDoubleScalarIsInf(dimension[maxdimlen])) {
        overflow = false;
        exitg2 = 1;
      } else {
        maxdimlen++;
      }
    } else {
      overflow = true;
      exitg2 = 1;
    }
  } while (exitg2 == 0);

  guard1 = false;
  if (overflow) {
    maxdimlen = 0;
    do {
      exitg1 = 0;
      if (maxdimlen < 2) {
        if ((-2.147483648E+9 > dimension[maxdimlen]) || (2.147483647E+9 <
             dimension[maxdimlen])) {
          overflow = false;
          exitg1 = 1;
        } else {
          maxdimlen++;
        }
      } else {
        overflow = true;
        exitg1 = 1;
      }
    } while (exitg1 == 0);

    if (overflow) {
      overflow = true;
    } else {
      guard1 = true;
    }
  } else {
    guard1 = true;
  }

  if (guard1) {
    overflow = false;
  }

  if (overflow) {
  } else {
    emlrtErrorWithMessageIdR2012b(&b_st, &xb_emlrtRTEI,
      "Coder:toolbox:eml_assert_valid_size_arg_invalidSizeVector", 4, 12,
      MIN_int32_T, 12, MAX_int32_T);
  }

  n = 1.0;
  for (maxdimlen = 0; maxdimlen < 2; maxdimlen++) {
    if (dimension[maxdimlen] <= 0.0) {
      n = 0.0;
    } else {
      n *= dimension[maxdimlen];
    }
  }

  if (2.147483647E+9 >= n) {
  } else {
    emlrtErrorWithMessageIdR2012b(&b_st, &wb_emlrtRTEI, "Coder:MATLAB:pmaxsize",
      0);
  }

  for (i10 = 0; i10 < 2; i10++) {
    sz[i10] = (int32_T)dimension[i10];
  }

  b_st.site = &wc_emlrtRSI;
  dimension[0] = x->size[0];
  maxdimlen = (int32_T)(uint32_T)dimension[0];
  if (1 > (int32_T)(uint32_T)dimension[0]) {
    maxdimlen = 1;
  }

  maxdimlen = muIntScalarMax_sint32(x->size[0], maxdimlen);
  if (sz[0] > maxdimlen) {
    b_st.site = &xc_emlrtRSI;
    error(&b_st);
  }

  if (sz[1] > maxdimlen) {
    b_st.site = &xc_emlrtRSI;
    error(&b_st);
  }

  for (i10 = 0; i10 < 2; i10++) {
    i11 = sz[i10];
    if (!(i11 >= 0)) {
      emlrtNonNegativeCheckR2012b(i11, &b_emlrtDCI, &st);
    }
  }

  i10 = value->size[0] * value->size[1];
  value->size[0] = sz[0];
  value->size[1] = sz[1];
  emxEnsureCapacity(&st, (emxArray__common *)value, i10, (int32_T)sizeof(real_T),
                    &s_emlrtRTEI);
  if (x->size[0] == value->size[0] * value->size[1]) {
  } else {
    emlrtErrorWithMessageIdR2012b(&st, &yb_emlrtRTEI,
      "Coder:MATLAB:getReshapeDims_notSameNumel", 0);
  }

  b_st.site = &yc_emlrtRSI;
  overflow = ((!(1 > x->size[0])) && (x->size[0] > 2147483646));
  if (overflow) {
    c_st.site = &cb_emlrtRSI;
    b_check_forloop_overflow_error(&c_st, true);
  }

  for (maxdimlen = 0; maxdimlen + 1 <= x->size[0]; maxdimlen++) {
    value->data[maxdimlen] = x->data[maxdimlen];
  }

  emxFree_real_T(&x);
  emlrtHeapReferenceStackLeaveFcnR2012b(sp);
}

void d_AbstNsoltCoefManipulator2d_lo(const emlrtStack *sp, const
  c_saivdr_dictionary_nsoltx_desi *obj, emxArray_real_T *arrayCoefs, uint32_T
  iCol, const emxArray_real_T *U)
{
  emxArray_int32_T *r14;
  uint32_T nRows_;
  uint32_T q0;
  uint32_T qY;
  uint64_T u15;
  uint32_T indexCol;
  int32_T b_arrayCoefs;
  int32_T i74;
  int32_T loop_ub;
  emxArray_real_T *b;
  int32_T i75;
  boolean_T innerDimOk;
  int32_T i76;
  emxArray_real_T *C;
  int32_T iv44[2];
  real_T alpha1;
  int32_T b_loop_ub;
  real_T beta1;
  char_T TRANSB;
  char_T TRANSA;
  ptrdiff_t m_t;
  ptrdiff_t n_t;
  ptrdiff_t k_t;
  ptrdiff_t lda_t;
  ptrdiff_t ldb_t;
  ptrdiff_t ldc_t;
  emlrtStack st;
  emlrtStack b_st;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b(sp);
  emxInit_int32_T1(sp, &r14, 1, &cb_emlrtRTEI, true);
  nRows_ = obj->nRows + MAX_uint32_T;
  q0 = iCol;
  qY = q0 - 1U;
  if (qY > q0) {
    qY = 0U;
  }

  u15 = (uint64_T)qY * (nRows_ - MAX_uint32_T);
  if (u15 > 4294967295ULL) {
    u15 = 4294967295ULL;
  }

  indexCol = (uint32_T)u15;
  b_arrayCoefs = arrayCoefs->size[1];
  i74 = r14->size[0];
  r14->size[0] = (int32_T)nRows_ + 1;
  emxEnsureCapacity(sp, (emxArray__common *)r14, i74, (int32_T)sizeof(int32_T),
                    &cb_emlrtRTEI);
  loop_ub = (int32_T)nRows_;
  for (i74 = 0; i74 <= loop_ub; i74++) {
    u15 = (uint64_T)indexCol + (1U + i74);
    if (u15 > 4294967295ULL) {
      u15 = 4294967295ULL;
    }

    i75 = (int32_T)u15;
    if (!((i75 >= 1) && (i75 <= b_arrayCoefs))) {
      emlrtDynamicBoundsCheckR2012b(i75, 1, b_arrayCoefs, &fb_emlrtBCI, sp);
    }

    r14->data[i74] = i75 - 1;
  }

  emxInit_real_T1(sp, &b, 2, &cb_emlrtRTEI, true);
  st.site = &be_emlrtRSI;
  b_arrayCoefs = arrayCoefs->size[1];
  i74 = b->size[0] * b->size[1];
  b->size[0] = 12;
  b->size[1] = (int32_T)nRows_ + 1;
  emxEnsureCapacity(&st, (emxArray__common *)b, i74, (int32_T)sizeof(real_T),
                    &cb_emlrtRTEI);
  loop_ub = (int32_T)nRows_;
  for (i74 = 0; i74 <= loop_ub; i74++) {
    for (i75 = 0; i75 < 12; i75++) {
      i76 = 1 + i74;
      if (i76 < 0) {
        i76 = 0;
      }

      u15 = (uint64_T)indexCol + i76;
      if (u15 > 4294967295ULL) {
        u15 = 4294967295ULL;
      }

      i76 = (int32_T)u15;
      if (!((i76 >= 1) && (i76 <= b_arrayCoefs))) {
        emlrtDynamicBoundsCheckR2012b(i76, 1, b_arrayCoefs, &gb_emlrtBCI, &st);
      }

      b->data[i75 + b->size[0] * i74] = arrayCoefs->data[(i75 + arrayCoefs->
        size[0] * (i76 - 1)) + 12];
    }
  }

  b_st.site = &ed_emlrtRSI;
  innerDimOk = (U->size[1] == 12);
  if (!innerDimOk) {
    if ((U->size[0] == 1) && (U->size[1] == 1)) {
      emlrtErrorWithMessageIdR2012b(&b_st, &sb_emlrtRTEI,
        "Coder:toolbox:mtimes_noDynamicScalarExpansion", 0);
    } else {
      emlrtErrorWithMessageIdR2012b(&b_st, &tb_emlrtRTEI,
        "Coder:MATLAB:innerdim", 0);
    }
  }

  emxInit_real_T1(&st, &C, 2, &cb_emlrtRTEI, true);
  if (U->size[1] == 1) {
    i74 = C->size[0] * C->size[1];
    C->size[0] = U->size[0];
    C->size[1] = b->size[1];
    emxEnsureCapacity(&st, (emxArray__common *)C, i74, (int32_T)sizeof(real_T),
                      &cb_emlrtRTEI);
    loop_ub = U->size[0];
    for (i74 = 0; i74 < loop_ub; i74++) {
      b_arrayCoefs = b->size[1];
      for (i75 = 0; i75 < b_arrayCoefs; i75++) {
        C->data[i74 + C->size[0] * i75] = 0.0;
        b_loop_ub = U->size[1];
        for (i76 = 0; i76 < b_loop_ub; i76++) {
          C->data[i74 + C->size[0] * i75] += U->data[i74 + U->size[0] * i76] *
            b->data[i76 + b->size[0] * i75];
        }
      }
    }
  } else {
    q0 = (uint32_T)U->size[0];
    i74 = C->size[0] * C->size[1];
    C->size[0] = (int32_T)q0;
    C->size[1] = (int32_T)nRows_ + 1;
    emxEnsureCapacity(&st, (emxArray__common *)C, i74, (int32_T)sizeof(real_T),
                      &cb_emlrtRTEI);
    loop_ub = C->size[1];
    for (i74 = 0; i74 < loop_ub; i74++) {
      b_arrayCoefs = C->size[0];
      for (i75 = 0; i75 < b_arrayCoefs; i75++) {
        C->data[i75 + C->size[0] * i74] = 0.0;
      }
    }

    b_st.site = &y_emlrtRSI;
    if ((U->size[0] < 1) || ((int32_T)nRows_ + 1 < 1) || (U->size[1] < 1)) {
    } else {
      alpha1 = 1.0;
      beta1 = 0.0;
      TRANSB = 'N';
      TRANSA = 'N';
      m_t = (ptrdiff_t)U->size[0];
      n_t = (ptrdiff_t)((int32_T)nRows_ + 1);
      k_t = (ptrdiff_t)U->size[1];
      lda_t = (ptrdiff_t)U->size[0];
      ldb_t = (ptrdiff_t)U->size[1];
      ldc_t = (ptrdiff_t)U->size[0];
      dgemm(&TRANSA, &TRANSB, &m_t, &n_t, &k_t, &alpha1, &U->data[0], &lda_t,
            &b->data[0], &ldb_t, &beta1, &C->data[0], &ldc_t);
    }
  }

  emxFree_real_T(&b);
  iv44[0] = 12;
  iv44[1] = r14->size[0];
  emlrtSubAssignSizeCheckR2012b(iv44, 2, *(int32_T (*)[2])C->size, 2,
    &n_emlrtECI, sp);
  b_arrayCoefs = r14->size[0];
  for (i74 = 0; i74 < b_arrayCoefs; i74++) {
    for (i75 = 0; i75 < 12; i75++) {
      arrayCoefs->data[(i75 + arrayCoefs->size[0] * r14->data[i74]) + 12] =
        C->data[i75 + 12 * i74];
    }
  }

  emxFree_real_T(&C);
  emxFree_int32_T(&r14);
  emlrtHeapReferenceStackLeaveFcnR2012b(sp);
}

void d_AbstNsoltCoefManipulator2d_se(const emlrtStack *sp,
  d_saivdr_dictionary_nsoltx_desi *obj)
{
  uint32_T ord[2];
  int32_T i21;
  emxArray_real_T *paramMtxSzTab_;
  real_T y;
  int32_T loop_ub;
  int32_T iOrd;
  emxArray_real_T *indexOfParamMtxSzTab_;
  real_T cidx;
  uint32_T b;
  uint32_T iRow;
  int32_T b_indexOfParamMtxSzTab_;
  emlrtHeapReferenceStackEnterFcnR2012b(sp);
  for (i21 = 0; i21 < 2; i21++) {
    ord[i21] = obj->PolyPhaseOrder[i21];
  }

  emxInit_real_T1(sp, &paramMtxSzTab_, 2, &o_emlrtRTEI, true);
  y = (real_T)ord[0] + (real_T)ord[1];
  i21 = paramMtxSzTab_->size[0] * paramMtxSzTab_->size[1];
  paramMtxSzTab_->size[0] = (int32_T)(y + 2.0);
  paramMtxSzTab_->size[1] = 2;
  emxEnsureCapacity(sp, (emxArray__common *)paramMtxSzTab_, i21, (int32_T)sizeof
                    (real_T), &n_emlrtRTEI);
  loop_ub = (int32_T)(y + 2.0) << 1;
  for (i21 = 0; i21 < loop_ub; i21++) {
    paramMtxSzTab_->data[i21] = 0.0;
  }

  i21 = (int32_T)(y + 2.0);
  if (!(1 <= i21)) {
    emlrtDynamicBoundsCheckR2012b(1, 1, i21, &o_emlrtBCI, sp);
  }

  for (i21 = 0; i21 < 2; i21++) {
    paramMtxSzTab_->data[paramMtxSzTab_->size[0] * i21] = 12.0;
  }

  i21 = paramMtxSzTab_->size[0];
  if (!(2 <= i21)) {
    emlrtDynamicBoundsCheckR2012b(2, 1, i21, &n_emlrtBCI, sp);
  }

  for (i21 = 0; i21 < 2; i21++) {
    paramMtxSzTab_->data[1 + paramMtxSzTab_->size[0] * i21] = 12.0;
  }

  y = (real_T)ord[0] + (real_T)ord[1];
  emlrtForLoopVectorCheckR2012b(1.0, 1.0, y, mxDOUBLE_CLASS, (int32_T)y,
    &vb_emlrtRTEI, sp);
  iOrd = 0;
  while (iOrd <= (int32_T)y - 1) {
    loop_ub = paramMtxSzTab_->size[0];
    i21 = (int32_T)((1.0 + (real_T)iOrd) + 2.0);
    if (!((i21 >= 1) && (i21 <= loop_ub))) {
      emlrtDynamicBoundsCheckR2012b(i21, 1, loop_ub, &m_emlrtBCI, sp);
    }

    for (i21 = 0; i21 < 2; i21++) {
      paramMtxSzTab_->data[((int32_T)((1.0 + (real_T)iOrd) + 2.0) +
                            paramMtxSzTab_->size[0] * i21) - 1] = 12.0;
    }

    iOrd++;
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b(sp);
    }
  }

  emxInit_real_T1(sp, &indexOfParamMtxSzTab_, 2, &p_emlrtRTEI, true);
  i21 = indexOfParamMtxSzTab_->size[0] * indexOfParamMtxSzTab_->size[1];
  indexOfParamMtxSzTab_->size[0] = paramMtxSzTab_->size[0];
  indexOfParamMtxSzTab_->size[1] = 3;
  emxEnsureCapacity(sp, (emxArray__common *)indexOfParamMtxSzTab_, i21, (int32_T)
                    sizeof(real_T), &n_emlrtRTEI);
  loop_ub = paramMtxSzTab_->size[0] * 3;
  for (i21 = 0; i21 < loop_ub; i21++) {
    indexOfParamMtxSzTab_->data[i21] = 0.0;
  }

  cidx = 1.0;
  b = (uint32_T)paramMtxSzTab_->size[0];
  iRow = 1U;
  while (iRow <= b) {
    i21 = paramMtxSzTab_->size[0];
    loop_ub = (int32_T)iRow;
    if (!((loop_ub >= 1) && (loop_ub <= i21))) {
      emlrtDynamicBoundsCheckR2012b(loop_ub, 1, i21, &l_emlrtBCI, sp);
    }

    b_indexOfParamMtxSzTab_ = indexOfParamMtxSzTab_->size[0];
    i21 = (int32_T)iRow;
    if (!((i21 >= 1) && (i21 <= b_indexOfParamMtxSzTab_))) {
      emlrtDynamicBoundsCheckR2012b(i21, 1, b_indexOfParamMtxSzTab_, &j_emlrtBCI,
        sp);
    }

    indexOfParamMtxSzTab_->data[(int32_T)iRow - 1] = cidx;
    for (i21 = 0; i21 < 2; i21++) {
      indexOfParamMtxSzTab_->data[((int32_T)iRow + indexOfParamMtxSzTab_->size[0]
        * (i21 + 1)) - 1] = paramMtxSzTab_->data[(loop_ub + paramMtxSzTab_->
        size[0] * i21) - 1];
    }

    i21 = paramMtxSzTab_->size[0];
    loop_ub = (int32_T)iRow;
    if (!((loop_ub >= 1) && (loop_ub <= i21))) {
      emlrtDynamicBoundsCheckR2012b(loop_ub, 1, i21, &k_emlrtBCI, sp);
    }

    cidx += paramMtxSzTab_->data[(int32_T)iRow - 1] * paramMtxSzTab_->data
      [((int32_T)iRow + paramMtxSzTab_->size[0]) - 1];
    iRow++;
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b(sp);
    }
  }

  emxFree_real_T(&paramMtxSzTab_);
  i21 = obj->indexOfParamMtxSzTab->size[0] * obj->indexOfParamMtxSzTab->size[1];
  obj->indexOfParamMtxSzTab->size[0] = indexOfParamMtxSzTab_->size[0];
  obj->indexOfParamMtxSzTab->size[1] = 3;
  emxEnsureCapacity(sp, (emxArray__common *)obj->indexOfParamMtxSzTab, i21,
                    (int32_T)sizeof(real_T), &n_emlrtRTEI);
  loop_ub = indexOfParamMtxSzTab_->size[0] * indexOfParamMtxSzTab_->size[1];
  for (i21 = 0; i21 < loop_ub; i21++) {
    obj->indexOfParamMtxSzTab->data[i21] = indexOfParamMtxSzTab_->data[i21];
  }

  emxFree_real_T(&indexOfParamMtxSzTab_);
  emlrtHeapReferenceStackLeaveFcnR2012b(sp);
}

void e_AbstNsoltCoefManipulator2d_ge(const emlrtStack *sp, const
  d_saivdr_dictionary_nsoltx_desi *obj, emxArray_real_T *value)
{
  int32_T i22;
  real_T startIdx;
  real_T dimension[2];
  real_T endIdx;
  int32_T maxdimlen;
  int32_T i23;
  emxArray_real_T *x;
  int32_T sz[2];
  int32_T exitg2;
  boolean_T overflow;
  boolean_T guard1 = false;
  int32_T exitg1;
  real_T n;
  emlrtStack st;
  emlrtStack b_st;
  emlrtStack c_st;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b(sp);
  i22 = obj->indexOfParamMtxSzTab->size[0];
  if (!(1 <= i22)) {
    emlrtDynamicBoundsCheckR2012b(1, 1, i22, &r_emlrtBCI, sp);
  }

  startIdx = obj->indexOfParamMtxSzTab->data[0];
  i22 = obj->indexOfParamMtxSzTab->size[0];
  if (!(1 <= i22)) {
    emlrtDynamicBoundsCheckR2012b(1, 1, i22, &q_emlrtBCI, sp);
  }

  for (i22 = 0; i22 < 2; i22++) {
    dimension[i22] = obj->indexOfParamMtxSzTab->data[obj->
      indexOfParamMtxSzTab->size[0] * (1 + i22)];
  }

  endIdx = (startIdx + dimension[0] * dimension[1]) - 1.0;
  if (startIdx > endIdx) {
    i22 = 0;
    maxdimlen = 0;
  } else {
    i22 = obj->paramMtxCoefs->size[0];
    if (startIdx != (int32_T)muDoubleScalarFloor(startIdx)) {
      emlrtIntegerCheckR2012b(startIdx, &emlrtDCI, sp);
    }

    i23 = (int32_T)startIdx;
    if (!((i23 >= 1) && (i23 <= i22))) {
      emlrtDynamicBoundsCheckR2012b(i23, 1, i22, &p_emlrtBCI, sp);
    }

    i22 = i23 - 1;
    i23 = obj->paramMtxCoefs->size[0];
    if (endIdx != (int32_T)muDoubleScalarFloor(endIdx)) {
      emlrtIntegerCheckR2012b(endIdx, &emlrtDCI, sp);
    }

    maxdimlen = (int32_T)endIdx;
    if (!((maxdimlen >= 1) && (maxdimlen <= i23))) {
      emlrtDynamicBoundsCheckR2012b(maxdimlen, 1, i23, &p_emlrtBCI, sp);
    }
  }

  emxInit_real_T(sp, &x, 1, &s_emlrtRTEI, true);
  sz[0] = 1;
  sz[1] = maxdimlen - i22;
  st.site = &sc_emlrtRSI;
  indexShapeCheck(&st, obj->paramMtxCoefs->size[0], sz);
  st.site = &tc_emlrtRSI;
  i23 = x->size[0];
  x->size[0] = maxdimlen - i22;
  emxEnsureCapacity(&st, (emxArray__common *)x, i23, (int32_T)sizeof(real_T),
                    &s_emlrtRTEI);
  maxdimlen -= i22;
  for (i23 = 0; i23 < maxdimlen; i23++) {
    x->data[i23] = obj->paramMtxCoefs->data[i22 + i23];
  }

  b_st.site = &vc_emlrtRSI;
  maxdimlen = 0;
  do {
    exitg2 = 0;
    if (maxdimlen < 2) {
      if ((dimension[maxdimlen] != muDoubleScalarFloor(dimension[maxdimlen])) ||
          muDoubleScalarIsInf(dimension[maxdimlen])) {
        overflow = false;
        exitg2 = 1;
      } else {
        maxdimlen++;
      }
    } else {
      overflow = true;
      exitg2 = 1;
    }
  } while (exitg2 == 0);

  guard1 = false;
  if (overflow) {
    maxdimlen = 0;
    do {
      exitg1 = 0;
      if (maxdimlen < 2) {
        if ((-2.147483648E+9 > dimension[maxdimlen]) || (2.147483647E+9 <
             dimension[maxdimlen])) {
          overflow = false;
          exitg1 = 1;
        } else {
          maxdimlen++;
        }
      } else {
        overflow = true;
        exitg1 = 1;
      }
    } while (exitg1 == 0);

    if (overflow) {
      overflow = true;
    } else {
      guard1 = true;
    }
  } else {
    guard1 = true;
  }

  if (guard1) {
    overflow = false;
  }

  if (overflow) {
  } else {
    emlrtErrorWithMessageIdR2012b(&b_st, &xb_emlrtRTEI,
      "Coder:toolbox:eml_assert_valid_size_arg_invalidSizeVector", 4, 12,
      MIN_int32_T, 12, MAX_int32_T);
  }

  n = 1.0;
  for (maxdimlen = 0; maxdimlen < 2; maxdimlen++) {
    if (dimension[maxdimlen] <= 0.0) {
      n = 0.0;
    } else {
      n *= dimension[maxdimlen];
    }
  }

  if (2.147483647E+9 >= n) {
  } else {
    emlrtErrorWithMessageIdR2012b(&b_st, &wb_emlrtRTEI, "Coder:MATLAB:pmaxsize",
      0);
  }

  for (i22 = 0; i22 < 2; i22++) {
    sz[i22] = (int32_T)dimension[i22];
  }

  b_st.site = &wc_emlrtRSI;
  dimension[0] = x->size[0];
  maxdimlen = (int32_T)(uint32_T)dimension[0];
  if (1 > (int32_T)(uint32_T)dimension[0]) {
    maxdimlen = 1;
  }

  maxdimlen = muIntScalarMax_sint32(x->size[0], maxdimlen);
  if (sz[0] > maxdimlen) {
    b_st.site = &xc_emlrtRSI;
    error(&b_st);
  }

  if (sz[1] > maxdimlen) {
    b_st.site = &xc_emlrtRSI;
    error(&b_st);
  }

  for (i22 = 0; i22 < 2; i22++) {
    i23 = sz[i22];
    if (!(i23 >= 0)) {
      emlrtNonNegativeCheckR2012b(i23, &b_emlrtDCI, &st);
    }
  }

  i22 = value->size[0] * value->size[1];
  value->size[0] = sz[0];
  value->size[1] = sz[1];
  emxEnsureCapacity(&st, (emxArray__common *)value, i22, (int32_T)sizeof(real_T),
                    &s_emlrtRTEI);
  if (x->size[0] == value->size[0] * value->size[1]) {
  } else {
    emlrtErrorWithMessageIdR2012b(&st, &yb_emlrtRTEI,
      "Coder:MATLAB:getReshapeDims_notSameNumel", 0);
  }

  b_st.site = &yc_emlrtRSI;
  overflow = ((!(1 > x->size[0])) && (x->size[0] > 2147483646));
  if (overflow) {
    c_st.site = &cb_emlrtRSI;
    b_check_forloop_overflow_error(&c_st, true);
  }

  for (maxdimlen = 0; maxdimlen + 1 <= x->size[0]; maxdimlen++) {
    value->data[maxdimlen] = x->data[maxdimlen];
  }

  emxFree_real_T(&x);
  emlrtHeapReferenceStackLeaveFcnR2012b(sp);
}

void f_AbstNsoltCoefManipulator2d_ge(const emlrtStack *sp, const
  d_saivdr_dictionary_nsoltx_desi *obj, emxArray_real_T *value)
{
  int32_T i24;
  real_T startIdx;
  real_T dimension[2];
  real_T endIdx;
  int32_T maxdimlen;
  int32_T i25;
  emxArray_real_T *x;
  int32_T sz[2];
  int32_T exitg2;
  boolean_T overflow;
  boolean_T guard1 = false;
  int32_T exitg1;
  real_T n;
  emlrtStack st;
  emlrtStack b_st;
  emlrtStack c_st;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b(sp);
  i24 = obj->indexOfParamMtxSzTab->size[0];
  if (!(2 <= i24)) {
    emlrtDynamicBoundsCheckR2012b(2, 1, i24, &r_emlrtBCI, sp);
  }

  startIdx = obj->indexOfParamMtxSzTab->data[1];
  i24 = obj->indexOfParamMtxSzTab->size[0];
  if (!(2 <= i24)) {
    emlrtDynamicBoundsCheckR2012b(2, 1, i24, &q_emlrtBCI, sp);
  }

  for (i24 = 0; i24 < 2; i24++) {
    dimension[i24] = obj->indexOfParamMtxSzTab->data[1 +
      obj->indexOfParamMtxSzTab->size[0] * (1 + i24)];
  }

  endIdx = (startIdx + dimension[0] * dimension[1]) - 1.0;
  if (startIdx > endIdx) {
    i24 = 0;
    maxdimlen = 0;
  } else {
    i24 = obj->paramMtxCoefs->size[0];
    if (startIdx != (int32_T)muDoubleScalarFloor(startIdx)) {
      emlrtIntegerCheckR2012b(startIdx, &emlrtDCI, sp);
    }

    i25 = (int32_T)startIdx;
    if (!((i25 >= 1) && (i25 <= i24))) {
      emlrtDynamicBoundsCheckR2012b(i25, 1, i24, &p_emlrtBCI, sp);
    }

    i24 = i25 - 1;
    i25 = obj->paramMtxCoefs->size[0];
    if (endIdx != (int32_T)muDoubleScalarFloor(endIdx)) {
      emlrtIntegerCheckR2012b(endIdx, &emlrtDCI, sp);
    }

    maxdimlen = (int32_T)endIdx;
    if (!((maxdimlen >= 1) && (maxdimlen <= i25))) {
      emlrtDynamicBoundsCheckR2012b(maxdimlen, 1, i25, &p_emlrtBCI, sp);
    }
  }

  emxInit_real_T(sp, &x, 1, &s_emlrtRTEI, true);
  sz[0] = 1;
  sz[1] = maxdimlen - i24;
  st.site = &sc_emlrtRSI;
  indexShapeCheck(&st, obj->paramMtxCoefs->size[0], sz);
  st.site = &tc_emlrtRSI;
  i25 = x->size[0];
  x->size[0] = maxdimlen - i24;
  emxEnsureCapacity(&st, (emxArray__common *)x, i25, (int32_T)sizeof(real_T),
                    &s_emlrtRTEI);
  maxdimlen -= i24;
  for (i25 = 0; i25 < maxdimlen; i25++) {
    x->data[i25] = obj->paramMtxCoefs->data[i24 + i25];
  }

  b_st.site = &vc_emlrtRSI;
  maxdimlen = 0;
  do {
    exitg2 = 0;
    if (maxdimlen < 2) {
      if ((dimension[maxdimlen] != muDoubleScalarFloor(dimension[maxdimlen])) ||
          muDoubleScalarIsInf(dimension[maxdimlen])) {
        overflow = false;
        exitg2 = 1;
      } else {
        maxdimlen++;
      }
    } else {
      overflow = true;
      exitg2 = 1;
    }
  } while (exitg2 == 0);

  guard1 = false;
  if (overflow) {
    maxdimlen = 0;
    do {
      exitg1 = 0;
      if (maxdimlen < 2) {
        if ((-2.147483648E+9 > dimension[maxdimlen]) || (2.147483647E+9 <
             dimension[maxdimlen])) {
          overflow = false;
          exitg1 = 1;
        } else {
          maxdimlen++;
        }
      } else {
        overflow = true;
        exitg1 = 1;
      }
    } while (exitg1 == 0);

    if (overflow) {
      overflow = true;
    } else {
      guard1 = true;
    }
  } else {
    guard1 = true;
  }

  if (guard1) {
    overflow = false;
  }

  if (overflow) {
  } else {
    emlrtErrorWithMessageIdR2012b(&b_st, &xb_emlrtRTEI,
      "Coder:toolbox:eml_assert_valid_size_arg_invalidSizeVector", 4, 12,
      MIN_int32_T, 12, MAX_int32_T);
  }

  n = 1.0;
  for (maxdimlen = 0; maxdimlen < 2; maxdimlen++) {
    if (dimension[maxdimlen] <= 0.0) {
      n = 0.0;
    } else {
      n *= dimension[maxdimlen];
    }
  }

  if (2.147483647E+9 >= n) {
  } else {
    emlrtErrorWithMessageIdR2012b(&b_st, &wb_emlrtRTEI, "Coder:MATLAB:pmaxsize",
      0);
  }

  for (i24 = 0; i24 < 2; i24++) {
    sz[i24] = (int32_T)dimension[i24];
  }

  b_st.site = &wc_emlrtRSI;
  dimension[0] = x->size[0];
  maxdimlen = (int32_T)(uint32_T)dimension[0];
  if (1 > (int32_T)(uint32_T)dimension[0]) {
    maxdimlen = 1;
  }

  maxdimlen = muIntScalarMax_sint32(x->size[0], maxdimlen);
  if (sz[0] > maxdimlen) {
    b_st.site = &xc_emlrtRSI;
    error(&b_st);
  }

  if (sz[1] > maxdimlen) {
    b_st.site = &xc_emlrtRSI;
    error(&b_st);
  }

  for (i24 = 0; i24 < 2; i24++) {
    i25 = sz[i24];
    if (!(i25 >= 0)) {
      emlrtNonNegativeCheckR2012b(i25, &b_emlrtDCI, &st);
    }
  }

  i24 = value->size[0] * value->size[1];
  value->size[0] = sz[0];
  value->size[1] = sz[1];
  emxEnsureCapacity(&st, (emxArray__common *)value, i24, (int32_T)sizeof(real_T),
                    &s_emlrtRTEI);
  if (x->size[0] == value->size[0] * value->size[1]) {
  } else {
    emlrtErrorWithMessageIdR2012b(&st, &yb_emlrtRTEI,
      "Coder:MATLAB:getReshapeDims_notSameNumel", 0);
  }

  b_st.site = &yc_emlrtRSI;
  overflow = ((!(1 > x->size[0])) && (x->size[0] > 2147483646));
  if (overflow) {
    c_st.site = &cb_emlrtRSI;
    b_check_forloop_overflow_error(&c_st, true);
  }

  for (maxdimlen = 0; maxdimlen + 1 <= x->size[0]; maxdimlen++) {
    value->data[maxdimlen] = x->data[maxdimlen];
  }

  emxFree_real_T(&x);
  emlrtHeapReferenceStackLeaveFcnR2012b(sp);
}

void g_AbstNsoltCoefManipulator2d_ge(const emlrtStack *sp, const
  d_saivdr_dictionary_nsoltx_desi *obj, uint32_T b_index, emxArray_real_T *value)
{
  int32_T i26;
  int32_T i27;
  real_T startIdx;
  int32_T maxdimlen;
  real_T dimension[2];
  real_T endIdx;
  emxArray_real_T *x;
  int32_T sz[2];
  int32_T exitg2;
  boolean_T overflow;
  boolean_T guard1 = false;
  int32_T exitg1;
  real_T n;
  emlrtStack st;
  emlrtStack b_st;
  emlrtStack c_st;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b(sp);
  i26 = obj->indexOfParamMtxSzTab->size[0];
  i27 = (int32_T)b_index;
  if (!((i27 >= 1) && (i27 <= i26))) {
    emlrtDynamicBoundsCheckR2012b(i27, 1, i26, &r_emlrtBCI, sp);
  }

  startIdx = obj->indexOfParamMtxSzTab->data[i27 - 1];
  i26 = obj->indexOfParamMtxSzTab->size[0];
  maxdimlen = (int32_T)b_index;
  if (!((maxdimlen >= 1) && (maxdimlen <= i26))) {
    emlrtDynamicBoundsCheckR2012b(maxdimlen, 1, i26, &q_emlrtBCI, sp);
  }

  for (i26 = 0; i26 < 2; i26++) {
    dimension[i26] = obj->indexOfParamMtxSzTab->data[(maxdimlen +
      obj->indexOfParamMtxSzTab->size[0] * (1 + i26)) - 1];
  }

  endIdx = (startIdx + dimension[0] * dimension[1]) - 1.0;
  if (startIdx > endIdx) {
    i26 = 0;
    maxdimlen = 0;
  } else {
    i26 = obj->paramMtxCoefs->size[0];
    if (startIdx != (int32_T)muDoubleScalarFloor(startIdx)) {
      emlrtIntegerCheckR2012b(startIdx, &emlrtDCI, sp);
    }

    i27 = (int32_T)startIdx;
    if (!((i27 >= 1) && (i27 <= i26))) {
      emlrtDynamicBoundsCheckR2012b(i27, 1, i26, &p_emlrtBCI, sp);
    }

    i26 = i27 - 1;
    i27 = obj->paramMtxCoefs->size[0];
    if (endIdx != (int32_T)muDoubleScalarFloor(endIdx)) {
      emlrtIntegerCheckR2012b(endIdx, &emlrtDCI, sp);
    }

    maxdimlen = (int32_T)endIdx;
    if (!((maxdimlen >= 1) && (maxdimlen <= i27))) {
      emlrtDynamicBoundsCheckR2012b(maxdimlen, 1, i27, &p_emlrtBCI, sp);
    }
  }

  emxInit_real_T(sp, &x, 1, &s_emlrtRTEI, true);
  sz[0] = 1;
  sz[1] = maxdimlen - i26;
  st.site = &sc_emlrtRSI;
  indexShapeCheck(&st, obj->paramMtxCoefs->size[0], sz);
  st.site = &tc_emlrtRSI;
  i27 = x->size[0];
  x->size[0] = maxdimlen - i26;
  emxEnsureCapacity(&st, (emxArray__common *)x, i27, (int32_T)sizeof(real_T),
                    &s_emlrtRTEI);
  maxdimlen -= i26;
  for (i27 = 0; i27 < maxdimlen; i27++) {
    x->data[i27] = obj->paramMtxCoefs->data[i26 + i27];
  }

  b_st.site = &vc_emlrtRSI;
  maxdimlen = 0;
  do {
    exitg2 = 0;
    if (maxdimlen < 2) {
      if ((dimension[maxdimlen] != muDoubleScalarFloor(dimension[maxdimlen])) ||
          muDoubleScalarIsInf(dimension[maxdimlen])) {
        overflow = false;
        exitg2 = 1;
      } else {
        maxdimlen++;
      }
    } else {
      overflow = true;
      exitg2 = 1;
    }
  } while (exitg2 == 0);

  guard1 = false;
  if (overflow) {
    maxdimlen = 0;
    do {
      exitg1 = 0;
      if (maxdimlen < 2) {
        if ((-2.147483648E+9 > dimension[maxdimlen]) || (2.147483647E+9 <
             dimension[maxdimlen])) {
          overflow = false;
          exitg1 = 1;
        } else {
          maxdimlen++;
        }
      } else {
        overflow = true;
        exitg1 = 1;
      }
    } while (exitg1 == 0);

    if (overflow) {
      overflow = true;
    } else {
      guard1 = true;
    }
  } else {
    guard1 = true;
  }

  if (guard1) {
    overflow = false;
  }

  if (overflow) {
  } else {
    emlrtErrorWithMessageIdR2012b(&b_st, &xb_emlrtRTEI,
      "Coder:toolbox:eml_assert_valid_size_arg_invalidSizeVector", 4, 12,
      MIN_int32_T, 12, MAX_int32_T);
  }

  n = 1.0;
  for (maxdimlen = 0; maxdimlen < 2; maxdimlen++) {
    if (dimension[maxdimlen] <= 0.0) {
      n = 0.0;
    } else {
      n *= dimension[maxdimlen];
    }
  }

  if (2.147483647E+9 >= n) {
  } else {
    emlrtErrorWithMessageIdR2012b(&b_st, &wb_emlrtRTEI, "Coder:MATLAB:pmaxsize",
      0);
  }

  for (i26 = 0; i26 < 2; i26++) {
    sz[i26] = (int32_T)dimension[i26];
  }

  b_st.site = &wc_emlrtRSI;
  dimension[0] = x->size[0];
  maxdimlen = (int32_T)(uint32_T)dimension[0];
  if (1 > (int32_T)(uint32_T)dimension[0]) {
    maxdimlen = 1;
  }

  maxdimlen = muIntScalarMax_sint32(x->size[0], maxdimlen);
  if (sz[0] > maxdimlen) {
    b_st.site = &xc_emlrtRSI;
    error(&b_st);
  }

  if (sz[1] > maxdimlen) {
    b_st.site = &xc_emlrtRSI;
    error(&b_st);
  }

  for (i26 = 0; i26 < 2; i26++) {
    i27 = sz[i26];
    if (!(i27 >= 0)) {
      emlrtNonNegativeCheckR2012b(i27, &b_emlrtDCI, &st);
    }
  }

  i26 = value->size[0] * value->size[1];
  value->size[0] = sz[0];
  value->size[1] = sz[1];
  emxEnsureCapacity(&st, (emxArray__common *)value, i26, (int32_T)sizeof(real_T),
                    &s_emlrtRTEI);
  if (x->size[0] == value->size[0] * value->size[1]) {
  } else {
    emlrtErrorWithMessageIdR2012b(&st, &yb_emlrtRTEI,
      "Coder:MATLAB:getReshapeDims_notSameNumel", 0);
  }

  b_st.site = &yc_emlrtRSI;
  overflow = ((!(1 > x->size[0])) && (x->size[0] > 2147483646));
  if (overflow) {
    c_st.site = &cb_emlrtRSI;
    b_check_forloop_overflow_error(&c_st, true);
  }

  for (maxdimlen = 0; maxdimlen + 1 <= x->size[0]; maxdimlen++) {
    value->data[maxdimlen] = x->data[maxdimlen];
  }

  emxFree_real_T(&x);
  emlrtHeapReferenceStackLeaveFcnR2012b(sp);
}

void h_AbstNsoltCoefManipulator2d_ge(const emlrtStack *sp, const
  c_saivdr_dictionary_nsoltx_desi *obj, uint32_T b_index, emxArray_real_T *value)
{
  int32_T i28;
  int32_T i29;
  real_T startIdx;
  int32_T maxdimlen;
  real_T dimension[2];
  real_T endIdx;
  emxArray_real_T *x;
  int32_T sz[2];
  int32_T exitg2;
  boolean_T overflow;
  boolean_T guard1 = false;
  int32_T exitg1;
  real_T n;
  emlrtStack st;
  emlrtStack b_st;
  emlrtStack c_st;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b(sp);
  i28 = obj->indexOfParamMtxSzTab->size[0];
  i29 = (int32_T)b_index;
  if (!((i29 >= 1) && (i29 <= i28))) {
    emlrtDynamicBoundsCheckR2012b(i29, 1, i28, &r_emlrtBCI, sp);
  }

  startIdx = obj->indexOfParamMtxSzTab->data[i29 - 1];
  i28 = obj->indexOfParamMtxSzTab->size[0];
  maxdimlen = (int32_T)b_index;
  if (!((maxdimlen >= 1) && (maxdimlen <= i28))) {
    emlrtDynamicBoundsCheckR2012b(maxdimlen, 1, i28, &q_emlrtBCI, sp);
  }

  for (i28 = 0; i28 < 2; i28++) {
    dimension[i28] = obj->indexOfParamMtxSzTab->data[(maxdimlen +
      obj->indexOfParamMtxSzTab->size[0] * (1 + i28)) - 1];
  }

  endIdx = (startIdx + dimension[0] * dimension[1]) - 1.0;
  if (startIdx > endIdx) {
    i28 = 0;
    maxdimlen = 0;
  } else {
    i28 = obj->paramMtxCoefs->size[0];
    if (startIdx != (int32_T)muDoubleScalarFloor(startIdx)) {
      emlrtIntegerCheckR2012b(startIdx, &emlrtDCI, sp);
    }

    i29 = (int32_T)startIdx;
    if (!((i29 >= 1) && (i29 <= i28))) {
      emlrtDynamicBoundsCheckR2012b(i29, 1, i28, &p_emlrtBCI, sp);
    }

    i28 = i29 - 1;
    i29 = obj->paramMtxCoefs->size[0];
    if (endIdx != (int32_T)muDoubleScalarFloor(endIdx)) {
      emlrtIntegerCheckR2012b(endIdx, &emlrtDCI, sp);
    }

    maxdimlen = (int32_T)endIdx;
    if (!((maxdimlen >= 1) && (maxdimlen <= i29))) {
      emlrtDynamicBoundsCheckR2012b(maxdimlen, 1, i29, &p_emlrtBCI, sp);
    }
  }

  emxInit_real_T(sp, &x, 1, &s_emlrtRTEI, true);
  sz[0] = 1;
  sz[1] = maxdimlen - i28;
  st.site = &sc_emlrtRSI;
  indexShapeCheck(&st, obj->paramMtxCoefs->size[0], sz);
  st.site = &tc_emlrtRSI;
  i29 = x->size[0];
  x->size[0] = maxdimlen - i28;
  emxEnsureCapacity(&st, (emxArray__common *)x, i29, (int32_T)sizeof(real_T),
                    &s_emlrtRTEI);
  maxdimlen -= i28;
  for (i29 = 0; i29 < maxdimlen; i29++) {
    x->data[i29] = obj->paramMtxCoefs->data[i28 + i29];
  }

  b_st.site = &vc_emlrtRSI;
  maxdimlen = 0;
  do {
    exitg2 = 0;
    if (maxdimlen < 2) {
      if ((dimension[maxdimlen] != muDoubleScalarFloor(dimension[maxdimlen])) ||
          muDoubleScalarIsInf(dimension[maxdimlen])) {
        overflow = false;
        exitg2 = 1;
      } else {
        maxdimlen++;
      }
    } else {
      overflow = true;
      exitg2 = 1;
    }
  } while (exitg2 == 0);

  guard1 = false;
  if (overflow) {
    maxdimlen = 0;
    do {
      exitg1 = 0;
      if (maxdimlen < 2) {
        if ((-2.147483648E+9 > dimension[maxdimlen]) || (2.147483647E+9 <
             dimension[maxdimlen])) {
          overflow = false;
          exitg1 = 1;
        } else {
          maxdimlen++;
        }
      } else {
        overflow = true;
        exitg1 = 1;
      }
    } while (exitg1 == 0);

    if (overflow) {
      overflow = true;
    } else {
      guard1 = true;
    }
  } else {
    guard1 = true;
  }

  if (guard1) {
    overflow = false;
  }

  if (overflow) {
  } else {
    emlrtErrorWithMessageIdR2012b(&b_st, &xb_emlrtRTEI,
      "Coder:toolbox:eml_assert_valid_size_arg_invalidSizeVector", 4, 12,
      MIN_int32_T, 12, MAX_int32_T);
  }

  n = 1.0;
  for (maxdimlen = 0; maxdimlen < 2; maxdimlen++) {
    if (dimension[maxdimlen] <= 0.0) {
      n = 0.0;
    } else {
      n *= dimension[maxdimlen];
    }
  }

  if (2.147483647E+9 >= n) {
  } else {
    emlrtErrorWithMessageIdR2012b(&b_st, &wb_emlrtRTEI, "Coder:MATLAB:pmaxsize",
      0);
  }

  for (i28 = 0; i28 < 2; i28++) {
    sz[i28] = (int32_T)dimension[i28];
  }

  b_st.site = &wc_emlrtRSI;
  dimension[0] = x->size[0];
  maxdimlen = (int32_T)(uint32_T)dimension[0];
  if (1 > (int32_T)(uint32_T)dimension[0]) {
    maxdimlen = 1;
  }

  maxdimlen = muIntScalarMax_sint32(x->size[0], maxdimlen);
  if (sz[0] > maxdimlen) {
    b_st.site = &xc_emlrtRSI;
    error(&b_st);
  }

  if (sz[1] > maxdimlen) {
    b_st.site = &xc_emlrtRSI;
    error(&b_st);
  }

  for (i28 = 0; i28 < 2; i28++) {
    i29 = sz[i28];
    if (!(i29 >= 0)) {
      emlrtNonNegativeCheckR2012b(i29, &b_emlrtDCI, &st);
    }
  }

  i28 = value->size[0] * value->size[1];
  value->size[0] = sz[0];
  value->size[1] = sz[1];
  emxEnsureCapacity(&st, (emxArray__common *)value, i28, (int32_T)sizeof(real_T),
                    &s_emlrtRTEI);
  if (x->size[0] == value->size[0] * value->size[1]) {
  } else {
    emlrtErrorWithMessageIdR2012b(&st, &yb_emlrtRTEI,
      "Coder:MATLAB:getReshapeDims_notSameNumel", 0);
  }

  b_st.site = &yc_emlrtRSI;
  overflow = ((!(1 > x->size[0])) && (x->size[0] > 2147483646));
  if (overflow) {
    c_st.site = &cb_emlrtRSI;
    b_check_forloop_overflow_error(&c_st, true);
  }

  for (maxdimlen = 0; maxdimlen + 1 <= x->size[0]; maxdimlen++) {
    value->data[maxdimlen] = x->data[maxdimlen];
  }

  emxFree_real_T(&x);
  emlrtHeapReferenceStackLeaveFcnR2012b(sp);
}

/* End of code generation (AbstNsoltCoefManipulator2d.c) */
