//---------------------------------------------------------------------------
#ifndef meqmathH
#define meqmathH

#include "deqsys.h"
#include "matrix.h"
#include "vector.h"

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))
/* Pi divided by 180 (for converting degrees to radians) */
extern const Extended M_PI_180;
/* 180 divided by Pi (for converting radians to degrees) */
extern const Extended M_180_PI;
//---------------------------------------------------------------------------
/* Vector to matrix conversion.
   Line=true - Result is a row matrix, otherwise a column matrix */
Matrix __fastcall VectorToMatrix(Vector V, bool Line = false);
/* Matrix to vector conversion.
   Line=true - Result is a row matrix, otherwise a column matrix */
Vector __fastcall MatrixToVector(Matrix M);
/* Returns a (3,3) rotation matrix around the Ox axis (Angle in radians) */
Matrix __fastcall RotX(Extended Angle);
/* Returns a (3,3) rotation matrix around the Oy axis (Angle in radians) */
Matrix __fastcall RotY(Extended Angle);
/* Returns a (3,3) rotation matrix around the Oz axis (Angle in radians) */
Matrix __fastcall RotZ(Extended Angle);
//---------------------------------------------------------------------------
#endif
