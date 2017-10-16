//---------------------------------------------------------------------------
#ifndef meqmathH
#define meqmathH

#include "matrix.h" 
#include "vector.h" 
#include "deqsys.h"

#define min(a, b)  (((a) < (b)) ? (a) : (b))
#define max(a, b)  (((a) > (b)) ? (a) : (b))  
/* ����� �� �������� �� 180 (��� �������� �������� � �������) */
extern const Extended M_PI_180;
/* 180 �������� �� ����� �� (��� �������� ������ � �������) */
extern const Extended M_180_PI;
//---------------------------------------------------------------------------
/* �������������� ������� � �������.
   Line=true - ��������� �������-������, ����� �������-������� */
Matrix __fastcall VectorToMatrix(Vector V, bool Line = false);
/* �������������� ������� � ������.
   Line=true - ��������� �������-������, ����� �������-������� */
Vector __fastcall MatrixToVector(Matrix M);
/* ���������� ������� (3,3) �������� ������ ��� Ox (���� Angle - � ���) */
Matrix __fastcall RotX(Extended Angle);
/* ���������� ������� (3,3) �������� ������ ��� Oy (���� Angle - � ���) */
Matrix __fastcall RotY(Extended Angle);
/* ���������� ������� (3,3) �������� ������ ��� Oz (���� Angle - � ���) */
Matrix __fastcall RotZ(Extended Angle);
//---------------------------------------------------------------------------
#endif
