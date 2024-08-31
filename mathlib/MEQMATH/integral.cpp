//---------------------------------------------------------------------------
#include <vcl.h>
#include <math.h>
#pragma hdrstop

#include "integral.h"
//---------------------------------------------------------------------------
__fastcall TIntPar::TIntPar()
{
F=NULL;
Abs=1e-6L;
Eps=1e-6L;
IntMeth=miSimpson;
RealEps=0.0L;
}
//---------------------------------------------------------------------------
AnsiString __fastcall TIntPar::GetErrStr()
{
char *ErrStr[4]={
/* 0 */"No errors",
/* 1 */"Unknown integration method",
/* 2 */"Incorrect accuracy specified",
/* 3 */"Too high accuracy specified, overflow occurred"
};
if((iErr<0)||(iErr>3))return AnsiString("Не правильный код ошибки");
return AnsiString(ErrStr[iErr]);
}
//---------------------------------------------------------------------------
Extended __fastcall Integral(TIntPar *IntPar)
{
switch(IntPar->IntMeth)
  {
  case miRectangle:
    return IntMethRect(IntPar->F,IntPar->A,IntPar->B,IntPar->Abs,IntPar->Eps,IntPar->iErr,IntPar->RealEps);
  case miTrapezoid:
    return IntMethTrap(IntPar->F,IntPar->A,IntPar->B,IntPar->Abs,IntPar->Eps,IntPar->iErr,IntPar->RealEps);
  case miSimpson:
    return IntMethSimp(IntPar->F,IntPar->A,IntPar->B,IntPar->Abs,IntPar->Eps,IntPar->iErr,IntPar->RealEps);
  default:
    IntPar->iErr=1;
  }
return 0.L;
}
//---------------------------------------------------------------------------
Extended __fastcall IntMethRect(HUnderIntFunc hf, const Extended& a,\
            const Extended& b, const Extended& abs, const Extended& eps,\
            int& iErr, Extended& Real)
{
bool Cond;
Cardinal m=1,i,lastm;
Extended h=(b-a)/m;
Extended x=a+0.5L*h;
Extended I=h*(*hf)(x);
if((eps<=0.0L)||(abs<=0.0L)){iErr=2;return I;}
Extended It;
do{
  It=I;
  lastm=m;
  m*=2;
  if(lastm>m){iErr=3;return I;}
  h=(b-a)/m;
  I=0.0L;
  x=a+0.5L*h;
  for(i=0;i<m;i++)
    {
    I+=h*(*hf)(x);
    x+=h;
    }
  Real=fabsl(I-It);
  Cond=(fabsl(I)<1.0L)?(Real>abs):(Real>fabsl(I*eps));
  }while(Cond);
iErr=0;
return I;
}
//---------------------------------------------------------------------------
Extended __fastcall IntMethTrap(HUnderIntFunc hf, const Extended& a, const Extended& b, const Extended& abs, const Extended& eps, int& iErr, Extended& Real)
{
}
//---------------------------------------------------------------------------
Extended __fastcall IntMethSimp(HUnderIntFunc hf, const Extended& a, const Extended& b, const Extended& abs, const Extended& eps, int& iErr, Extended& Real)
{
}
//---------------------------------------------------------------------------
#pragma package(smart_init)
