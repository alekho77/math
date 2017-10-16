/*
    �������� ������� ���� "������"
    08.01.2000
    (c) ����� ������� �. (Doctor Alex)
*/
//---------------------------------------------------------------------------
#include <vcl.h>
#include <math.h>
#pragma hdrstop

#include "vector.h"
#include "matrix.h"
Matrix __fastcall VectorToMatrix(Vector V, bool Line = false);
Vector __fastcall MatrixToVector(Matrix M);
//---------------------------------------------------------------------------
#pragma package(smart_init)

#define MaxVectorErrCode    10
TVectorError *VectorError = new TVectorError;
//---------------------------------------------------------------------------
/*  ������������� ������, Dim �� ���������� */
void __fastcall Vector::Free(void)
{
if(V!=NULL)delete V;
V=NULL;
}
//---------------------------------------------------------------------------
/* ���� Dim �� ����� ����, ������� ������ Dim ����� */
void __fastcall Vector::TryCreate(void)
{
if(Dim)
  {
  V=new Extended[Dim];
  if(V==NULL)VectorError->Add(1); // �������� ������
  }
}
//---------------------------------------------------------------------------
void __fastcall Vector::SetDim(const int Val)
{
if((Val>=0)&&(FDim!=Val))
  {
  Free();
  FDim=Val;
  TryCreate();
  }
}
//---------------------------------------------------------------------------
__fastcall Vector::Vector()
{
FDim=0;
V=NULL;
}
//---------------------------------------------------------------------------
__fastcall Vector::Vector(const int D)
{
FDim=0;
V=NULL;
Dim=D;
}
//---------------------------------------------------------------------------
__fastcall Vector::Vector(const Vector& VSrc)
{
FDim=0;
V=NULL;
*this=VSrc;
}
//---------------------------------------------------------------------------
__fastcall Vector::~Vector()
{
Free();
}
//---------------------------------------------------------------------------
    /* �������� ����� (���������� ���������� � ������� ��������) */
Vector& __fastcall Vector::operator = (const Vector& VSrc)
{
Dim=VSrc.Dim;
int i;
if(VSrc.V!=NULL)
  for(i=0;i<Dim;i++) V[i]=VSrc.V[i];
return *this;
}
//---------------------------------------------------------------------------
    /* ������������� ������� �������� �����
       (���� Src==NULL ������������� ���������� ������, � Dim=0)
       ���������� ������� ������ ������ */
Vector& __fastcall Vector::operator = (const Extended *Src)
{
if(Src==NULL)
  {
  Free();
  FDim=0;
  return *this;
  }
if(!Dim)
  {
  VectorError->Add(2); // ����������� ������� �� ����������
  return *this;
  }
int i;
for(i=0;i<Dim;i++) V[i]=Src[i];
return *this;
}
//---------------------------------------------------------------------------
    /* ���������� ������� ������ ������ */
Vector& __fastcall Vector::operator = (const Extended Initial)
{
if(!Dim)
  {
  VectorError->Add(2); // ����������� ������� �� ����������
  return *this;
  }
int i;
for(i=0;i<Dim;i++) V[i]=Initial;
return *this;
}
//---------------------------------------------------------------------------
    /* ����������� �������� ������ ��������� */
Vector& __fastcall Vector::operator += (const Vector& VSum)
{
if((V==NULL)||(VSum.V==NULL))
  {
  VectorError->Add(3); // ������ ��������� �� ���������
  return *this;
  }
if(Dim!=VSum.Dim)
  {
  VectorError->Add(4); // �������������� ����� ��������� ��������
  return *this;
  }
int i;
for(i=0;i<Dim;i++) V[i]+=VSum.V[i];
return *this;
}
//---------------------------------------------------------------------------
    /* ����������� �������� ������ ��������� */
Vector __fastcall Vector::operator + (const Vector& VS) const
{
Vector VSum(*this);
VSum+=VS;
return VSum;
}
//---------------------------------------------------------------------------
    /* ����������� �������� ������ ��������� */
Vector& __fastcall Vector::operator -= (const Vector& VSum)
{
if((V==NULL)||(VSum.V==NULL))
  {
  VectorError->Add(3); // ������ ��������� �� ���������
  return *this;
  }
if(Dim!=VSum.Dim)
  {
  VectorError->Add(4); // �������������� ����� ��������� ��������
  return *this;
  }
int i;
for(i=0;i<Dim;i++) V[i]-=VSum.V[i];
return *this;
}
//---------------------------------------------------------------------------
    /* ����������� �������� ������ ��������� */
Vector __fastcall Vector::operator - (const Vector& VS) const
{
Vector VSum(*this);
VSum-=VS;
return VSum;
}
//---------------------------------------------------------------------------
Vector& __fastcall Vector::operator *= (const Extended& Val)
{
if(V==NULL)
  {
  VectorError->Add(5); // ������ ��������� �� ���������
  return *this;
  }
int i;
for(i=0;i<Dim;i++) V[i]*=Val;
return *this;
}
//---------------------------------------------------------------------------
Vector& __fastcall Vector::operator /= (const Extended& Val)
{
if(V==NULL)
  {
  VectorError->Add(5); // ������ ��������� �� ���������
  return *this;
  }
int i;
for(i=0;i<Dim;i++) V[i]/=Val;
return *this;
}
//---------------------------------------------------------------------------
Vector __fastcall operator * (const Extended& Val, const Vector& VMul)
{
Vector VRez(VMul);
VRez*=Val;
return VRez;
}
//---------------------------------------------------------------------------
Vector __fastcall Vector::operator * (const Extended& Val) const
{
Vector VRez(*this);
VRez*=Val;
return VRez;
}
//---------------------------------------------------------------------------
Vector __fastcall Vector::operator / (const Extended& Val) const
{
Vector VRez(*this);
VRez/=Val;
return VRez;
}
//---------------------------------------------------------------------------
    /* ��������� ���������. ����������� �������� ������ ���������. */
Extended __fastcall Vector::operator * (const Vector& VMul) const
{
if((V==NULL)||(VMul.V==NULL))
  {
  VectorError->Add(5); // ������ ��������� �� ���������
  return 0.0L;
  }
if(Dim!=VMul.Dim)
  {
  VectorError->Add(6); // �������������� ����� ��������� �������� ����������
  return 0.0L;
  }
Extended Rez=0.0L;
int i;
for(i=0;i<Dim;i++)
  Rez+=V[i]*VMul.V[i];
return Rez;
}
//---------------------------------------------------------------------------
    /* ��������� ������� �� ������. ������� : M(m,n)*V(n) = V(m) */
Vector __fastcall operator * (const Matrix& M, const Vector& V)
{
return MatrixToVector(M*VectorToMatrix(V));
}
//---------------------------------------------------------------------------
Vector __fastcall Vector::operator - (void) const
{
if(V==NULL)
  {
  VectorError->Add(7); // ������ �� ����������
  return *this;
  }
Vector VRez(Dim);
int i;
for(i=0;i<Dim;i++) VRez.V[i]=-V[i];
return VRez;
}
//---------------------------------------------------------------------------
    /* ��������� ��������� */
    /* ��������� ������, ���� : ������� ��������� � ����� ����� ����� ��������������� �������� */
bool __fastcall Vector::operator == (const Vector& VCom) const
{
if((V==NULL)||(VCom.V==NULL))
  {
  VectorError->Add(8); // ������� �� ����������
  return false;
  }
if(Dim!=VCom.Dim) return false;
bool Rez;
int i;
for(i=0;i<Dim;i++)
  {
  Rez=(V[i]==VCom.V[i]);
  if(!Rez)break;
  }
return Rez;
}
//---------------------------------------------------------------------------
bool __fastcall Vector::operator != (const Vector& VCom) const
{
return !((*this)==VCom);
}
//---------------------------------------------------------------------------
        /* ��������� ������, ���� : ��� �������� ����� ��������� ����� */
bool __fastcall Vector::operator == (const Extended& Val) const
{
if(V==NULL)
  {
  VectorError->Add(8); // ������� �� ����������
  return false;
  }
bool Rez;
int i;
for(i=0;i<Dim;i++)
  {
  Rez=(V[i]==Val);
  if(!Rez)break;
  }
return Rez;
}
//---------------------------------------------------------------------------
bool __fastcall Vector::operator != (const Extended& Val) const
{
return !((*this)==Val);
}
//---------------------------------------------------------------------------
    /* ��������� ��������. Count = 1..Dim */
Extended& __fastcall Vector::operator [] (int Count)
{
if(Count<=0)
  {
  VectorError->Add(9); // ����������� �������� ������
  Count=1;
  }
else if(Count>Dim)
       {
       VectorError->Add(9); // ����������� �������� ������
       Count=Dim;
       }
return V[Count-1];
}
//---------------------------------------------------------------------------
    /* ����������� ������ ������� */
Extended __fastcall Vector::Mod(void)
{
Extended Rez;
Rez=(*this)*(*this);
return sqrtl(Rez);
}
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
    /* ���������� 0, ���� ������ ���� */
int __fastcall TVectorError::GetLastError(void)
{
if(Count)return iErr[Count-1];
return Count;
}
//---------------------------------------------------------------------------
__fastcall TVectorError::TVectorError()
{
FCount=0;
}
//---------------------------------------------------------------------------
__fastcall TVectorError::~TVectorError()
{
if(Count)delete iErr;
}
//---------------------------------------------------------------------------
    /* ����� �������� ������ � ��������� ������, ��� ������� ��������� � ������� ��� ������� Index */
    /* Index ������� �������� = 1 */
AnsiString __fastcall TVectorError::ErrStr(int Index)
{
static char *VectorErrStr[MaxVectorErrCode]={
/* 0*/ "TVectorError::ErrStr() : �� ����������� ������� ������.",
/* 1*/ "Vector::Create() : �� ������� ������ ��� �������.",
/* 2*/ "Vector::operator = : ������������� ����������, ��� ��� ����������� ������� �� ����������.",
/* 3*/ "Vector::operator +(-) : ������ ��������� �� ���������.",
/* 4*/ "Vector::operator +(-) : �������������� ����� ��������� ��������."
/* 5*/ "Vector::operator *(/) : ������ ��������� �� ���������.",
/* 6*/ "Vector::operator * : �������������� ����� ��������� �������� ����������.",
/* 7*/ "������� Vector::operator - : ������ �� ����������.",
/* 8*/ "Vector::operator == : ������� �� ����������.",
/* 9*/ "Vector::operator [] : ����������� �������� ������"
};
if((Index<=0)||(Index>Count))Index=0;
else Index=iErr[Index-1];
return AnsiString(VectorErrStr[Index]);
}
//---------------------------------------------------------------------------
    /* ������� ������ � �������� */
void __fastcall TVectorError::Clear(void)
{
if(Count)delete iErr;
FCount=0;
}
//---------------------------------------------------------------------------
    /* �������� ������ */
void __fastcall TVectorError::Add(int ErrCode)
{
if((ErrCode<1)||(ErrCode>=MaxVectorErrCode))return;
int *iErrt=new int[Count+1];
int i;
for(i=0;i<Count;i++)iErrt[i]=iErr[i];
if(Count)delete iErr;
iErr=iErrt;
iErr[Count]=ErrCode;
FCount++;
}
//---------------------------------------------------------------------------

