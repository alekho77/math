/*
    �������� ������� ���� "�������"
    08.01.2000
    (c) ����� ������� �. (Doctor Alex)
*/
//---------------------------------------------------------------------------
#include <vcl.h>
#include <stdio.h>
#pragma hdrstop

#include "matrix.h"
//---------------------------------------------------------------------------
#pragma package(smart_init)

#define MaxMatrixErrCode    10
TMatrixError *MatrixError = new TMatrixError;
//---------------------------------------------------------------------------
        /* ������������� ������, Col � Lin �� ���������� */
void __fastcall Matrix::Free(void)
{
if(M!=NULL)delete M;
M=NULL;
}
//---------------------------------------------------------------------------
        /* ���� Col � Lin �� ����� ����, ������� ������ (Col*Lin) ����� */
void __fastcall Matrix::TryCreate(void)
{
if(Col&&Lin)
  {
  M=new Extended[Col*Lin];
  if(M==NULL)MatrixError->Add(1);
  }
}
//---------------------------------------------------------------------------
void __fastcall Matrix::SetCol(const int Val)
{
if((Val>=0)&&(FCol!=Val))
  {
  Free();
  FCol=Val;
  TryCreate();
  }
}
//---------------------------------------------------------------------------
void __fastcall Matrix::SetLin(const int Val)
{
if((Val>=0)&&(FLin!=Val))
  {
  Free();
  FLin=Val;
  TryCreate();
  }
}
//---------------------------------------------------------------------------
__fastcall Matrix::Matrix()
{
FCol=FLin=0;
M=NULL;
}
//---------------------------------------------------------------------------
__fastcall Matrix::Matrix(const int L, const int C)
{
FCol=FLin=0;
M=NULL;
Lin=L;
Col=C;
}
//---------------------------------------------------------------------------
__fastcall Matrix::Matrix(const Matrix& MSrc)
{
FCol=FLin=0;
M=NULL;
*this=MSrc;
}
//---------------------------------------------------------------------------
__fastcall Matrix::~Matrix()
{
Free();
}
//---------------------------------------------------------------------------
    /* �������� ����� (���������� ���������� � ������� ��������) */
    /* ���� ������� �������� �����, �� ���������� ������� ����� ������ */
Matrix& __fastcall Matrix::operator = (const Matrix& MSrc)
{
Col=MSrc.Col;
Lin=MSrc.Lin;
int i;
if(MSrc.M!=NULL)
  for(i=0;i<Lin*Col;i++) M[i]=MSrc.M[i];
return *this;
}
//---------------------------------------------------------------------------
    /* ������������� ������� �������� �����.
       ���� Src==NULL ������������� ���������� ������, � Lin=Col=0.
       ���������� ������� ������ �������. */
Matrix& __fastcall Matrix::operator = (const Extended *Src)
{
if(Src==NULL)
  {
  Free();
  FCol=FLin=0;
  return *this;
  }
if(!(Col&&Lin))
  {
  MatrixError->Add(2); // ������ ������� �� ���������
  return *this;
  }
int i;
for(i=0;i<Lin*Col;i++) M[i]=Src[i];
return *this;
}
//---------------------------------------------------------------------------
    /* ������������� ������. ���������� ������� ������ ������� */
Matrix& __fastcall Matrix::operator = (const Extended Initial)
{
if(!(Col&&Lin))
  {
  MatrixError->Add(2); // ������ ������� �� ���������
  return *this;
  }
int i;
for(i=0;i<Lin*Col;i++) M[i]=Initial;
return *this;
}
//---------------------------------------------------------------------------
Matrix& __fastcall Matrix::operator += (const Matrix& MSum)
{
if((M==NULL)||(MSum.M==NULL))
  {
  MatrixError->Add(3); // ������� ��������� �� ����������
  return *this;
  }
if((Lin!=MSum.Lin)||(Col!=MSum.Col))
  {
  MatrixError->Add(4); // �������������� ����� ��������� ������
  return *this;
  }
int i;
for(i=0;i<Lin*Col;i++) M[i]+=MSum.M[i];
return *this;
}
//---------------------------------------------------------------------------
Matrix __fastcall Matrix::operator + (const Matrix& MS) const
{
Matrix MSum(*this);
MSum+=MS;
return MSum;
}
//---------------------------------------------------------------------------
Matrix& __fastcall Matrix::operator -= (const Matrix& MSum)
{
if((M==NULL)||(MSum.M==NULL))
  {
  MatrixError->Add(3); // ������� ��������� �� ����������
  return *this;
  }
if((Lin!=MSum.Lin)||(Col!=MSum.Col))
  {
  MatrixError->Add(4); // �������������� ����� ��������� ������
  return *this;
  }
int i;
for(i=0;i<Lin*Col;i++) M[i]-=MSum.M[i];
return *this;
}
//---------------------------------------------------------------------------
Matrix __fastcall Matrix::operator - (const Matrix& MS) const
{
Matrix MSum(*this);
MSum-=MS;
return MSum;
}
//---------------------------------------------------------------------------
Matrix& __fastcall Matrix::operator *= (const Extended& Val)
{
if(M==NULL)
  {
  MatrixError->Add(5); // ������� ��������� �� ����������
  return *this;
  }
int i;
for(i=0;i<Lin*Col;i++) M[i]*=Val;
return *this;
}
//---------------------------------------------------------------------------
Matrix& __fastcall Matrix::operator /= (const Extended& Val)
{
if(M==NULL)
  {
  MatrixError->Add(5); // ������� ��������� �� ����������
  return *this;
  }
int i;
for(i=0;i<Lin*Col;i++) M[i]/=Val;
return *this;
}
//---------------------------------------------------------------------------
Matrix __fastcall operator * (const Extended& Val, const Matrix& MMul)
{
Matrix MRez(MMul);
MRez*=Val;
return MRez;
}
//---------------------------------------------------------------------------
Matrix __fastcall Matrix::operator * (const Extended& Val) const
{
Matrix MRez(*this);
MRez*=Val;
return MRez;
}
//---------------------------------------------------------------------------
Matrix __fastcall Matrix::operator / (const Extended& Val) const
{
Matrix MRez(*this);
MRez/=Val;
return MRez;
}
//---------------------------------------------------------------------------
    /* ����������� ������ : M1(l,m)*M2(m,n)=M3(l,n) */
Matrix& __fastcall Matrix::operator *= (const Matrix& MMul)
{
(*this)=(*this)*MMul;
return *this;
}
//---------------------------------------------------------------------------
    /* ����������� ������ : M1(l,m)*M2(m,n)=M3(l,n) */
Matrix __fastcall Matrix::operator * (const Matrix& MMul) const
{
if((M==NULL)||(MMul.M==NULL))
  {
  MatrixError->Add(5); // ������� ��������� �� ����������
  return *this;
  }
if(Col!=MMul.Lin)
  {
  MatrixError->Add(6); // �������������� ����� ��������� ������ ����������
  return *this;
  }
Matrix MRez(Lin,MMul.Col);
int i,j,k;
for(i=0;i<Lin;i++)
  for(j=0;j<MMul.Col;j++)
    {
    MRez.M[i*MMul.Col+j]=0.0L;
    for(k=0;k<Col;k++)
      MRez.M[i*MMul.Col+j]+=M[i*Col+k]*MMul.M[k*MMul.Col+j];
    }
return MRez;
}
//---------------------------------------------------------------------------
Matrix __fastcall Matrix::operator - (void) const
{
if(M==NULL)
  {
  MatrixError->Add(7); // ������� �� ����������
  return *this;
  }
Matrix MRez(*this);
int i;
for(i=0;i<Lin*Col;i++) MRez.M[i]=-MRez.M[i];
return MRez;
}
//---------------------------------------------------------------------------
    /* ��������� ������ ���� :
        ������� ��������� � ����� ����� ����� ��������������� �������� */
bool __fastcall Matrix::operator == (const Matrix& MCom) const
{
if((M==NULL)||(MCom.M==NULL))
  {
  MatrixError->Add(8); // ������� �� ����������
  return false;
  }
if((Col!=MCom.Col)||(Lin!=MCom.Lin)) return false;
bool Rez;
int i;
for(i=0;i<Lin*Col;i++)
  {
  Rez=(M[i]==MCom.M[i]);
  if(!Rez)break;
  }
return Rez;
}
//---------------------------------------------------------------------------
bool __fastcall Matrix::operator != (const Matrix& MCom) const
{
return !((*this)==MCom);
}
//---------------------------------------------------------------------------
        /* ��������� ������, ���� : ��� �������� ����� ��������� ����� */
bool __fastcall Matrix::operator == (const Extended& Val) const
{
if(M==NULL)
  {
  MatrixError->Add(8); // ������� �� ����������
  return false;
  }
bool Rez;
int i;
for(i=0;i<Lin*Col;i++)
  {
  Rez=(M[i]==Val);
  if(!Rez)break;
  }
return Rez;
}
//---------------------------------------------------------------------------
bool __fastcall Matrix::operator != (const Extended& Val) const
{
return !((*this)==Val);
}
//---------------------------------------------------------------------------
    /* ��������� ��������. Count = 1..(Lin*Col) */
Extended& __fastcall Matrix::operator [] (int Count)
{
if(Count<=0)
  {
  MatrixError->Add(9); // ����������� �������� ������
  Count=1;
  }
else if(Count>Lin*Col)
       {
       MatrixError->Add(9); // ����������� �������� ������
       Count=Lin*Col;
       }
return M[Count-1];
}
//---------------------------------------------------------------------------
    /* ������ ��� ����������� ������ �������� �� ������ ������ � ������� */
int __fastcall Matrix::Index(int L, int C)
{
return (L-1)*Col+C;
}
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
    /* ���������� 0, ���� ������ ���� */
int __fastcall TMatrixError::GetLastError(void)
{
if(Count)return iErr[Count-1];
return Count;
}
//---------------------------------------------------------------------------
__fastcall TMatrixError::TMatrixError()
{
FCount=0;
}
//---------------------------------------------------------------------------
__fastcall TMatrixError::~TMatrixError()
{
if(Count)delete iErr;
}
//---------------------------------------------------------------------------
    /* ����� �������� ������ � ��������� ������, ��� ������� ��������� � ������� ��� ������� Index */
    /* Index ������� �������� = 1 */
char *MatrixErrStr[MaxMatrixErrCode]={
/* 0*/ "TMatrixError::ErrStr() : �� ����������� ������� ������.",
/* 1*/ "Matrix::Create() : �� ������� ������ ��� �������.",
/* 2*/ "Matrix::operator = : ������������� ����������, ��� ��� ������ ������� �� ���������.",
/* 3*/ "Matrix::operator +(-) : ������� ��������� �� ����������.",
/* 4*/ "Matrix::operator +(-) : �������������� ����� ��������� ������ ���������."
/* 5*/ "Matrix::operator *(/) : ������� ��������� �� ����������.",
/* 6*/ "Matrix::operator * : �������������� ����� ��������� ������ ����������.",
/* 7*/ "������� Matrix::operator - : ������� �� ����������.",
/* 8*/ "Matrix::operator == : ������� �� ����������.",
/* 9*/ "Matrix::operator [] : ����������� �������� ������"
};
AnsiString __fastcall TMatrixError::ErrStr(int Index)
{
if((Index<=0)||(Index>Count))Index=0;
else Index=iErr[Index-1];
return AnsiString(MatrixErrStr[Index]);
}
//---------------------------------------------------------------------------
    /* ������� ������ � �������� */
void __fastcall TMatrixError::Clear(void)
{
if(Count)delete iErr;
FCount=0;
}
//---------------------------------------------------------------------------
    /* �������� ������ */
void __fastcall TMatrixError::Add(int ErrCode)
{
if((ErrCode<1)||(ErrCode>=MaxMatrixErrCode))return;
int *iErrt=new int[Count+1];
int i;
for(i=0;i<Count;i++)iErrt[i]=iErr[i];
if(Count)delete iErr;
iErr=iErrt;
iErr[Count]=ErrCode;
FCount++;
}
//---------------------------------------------------------------------------

