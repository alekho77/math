//---------------------------------------------------------------------------
#include <vcl.h>
#include <math.h>
#pragma hdrstop

#include "meqmath.h"
//---------------------------------------------------------------------------
#pragma package(smart_init)

/* Pi divided by 180 (for converting degrees to radians) */
const Extended M_PI_180 = 0.0174532925199432957692369076848861L;
/* 180 divided by Pi (for converting radians to degrees) */
const Extended M_180_PI = 57.2957795130823208767981548141052L;
//---------------------------------------------------------------------------
/* Vector to matrix conversion.
   Line=true - Result is a row matrix, otherwise a column matrix */
Matrix __fastcall VectorToMatrix(Vector V, bool Line)
{
Matrix M;
if(!(V.Dim))return M;
if(Line)
  {
  M.Lin=1;
  M.Col=V.Dim;
  }
else
  {
  M.Lin=V.Dim;
  M.Col=1;
  }
for(int i=1;i<=V.Dim;i++) M[i]=V[i];
return M;
}
//---------------------------------------------------------------------------
/* Matrix to vector conversion.
   Line=true - Result is a row matrix, otherwise a column matrix */
Vector __fastcall MatrixToVector(Matrix M)
{
Vector V;
if(!((M.Lin)&&(M.Col)))return V;
V.Dim=(M.Lin)*(M.Col);
for(int i=1;i<=V.Dim;i++)V[i]=M[i];
return V;
}
//---------------------------------------------------------------------------
/* Returns a (3,3) rotation matrix around the Ox axis (Angle in radians) */
Matrix __fastcall RotX(Extended Angle)
{
Matrix Rx(3,3);
Extended cosFi=cosl(Angle);
Extended sinFi=sinl(Angle);
Rx=0.0L;
Rx[1]=1.0L;
Rx[5]=cosFi;
Rx[6]=sinFi;
Rx[8]=-sinFi;
Rx[9]=cosFi;
return Rx;
}
//---------------------------------------------------------------------------
/* Returns a (3,3) rotation matrix around the Oy axis (Angle in radians) */
Matrix __fastcall RotY(Extended Angle)
{
Matrix Ry(3,3);
Extended cosPsi=cosl(Angle);
Extended sinPsi=sinl(Angle);
Ry=0.0L;
Ry[1]=cosPsi;
Ry[3]=-sinPsi;
Ry[5]=1.0L;
Ry[7]=sinPsi;
Ry[9]=cosPsi;
return Ry;
}
//---------------------------------------------------------------------------
/* Returns a (3,3) rotation matrix around the Oz axis (Angle in radians) */
Matrix __fastcall RotZ(Extended Angle)
{
Matrix Rz(3,3);
Extended cosHi=cosl(Angle);
Extended sinHi=sinl(Angle);
Rz=0.0L;
Rz[1]=cosHi;
Rz[2]=sinHi;
Rz[4]=-sinHi;
Rz[5]=cosHi;
Rz[9]=1.0L;
return Rz;
}
//---------------------------------------------------------------------------

