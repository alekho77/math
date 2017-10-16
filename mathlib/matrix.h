/*
    Object "Matrix" for linear algebra.
    (c) 2017 Aleksey Khozin
*/

#ifndef MATHLIB_MATRIX_H
#define MATHLIB_MATRIX_H

namespace mathlib {

template <typename T>
class matrix
{
public:
  matrix() = default;
  matrix(const int r, const int c, const T& init = T()) {

  }
  //matrix(const Matrix& MSrc);
  //~matrix();

  int cols() const;  // number of columns
  int rows() const;  // number of rows

  //Matrix& __fastcall operator = (const Matrix& MSrc);
  /* ������������� ������� �������� ����� (���� Src==NULL ������������� ���������� ������, � Lin=Col=0) */
  //Matrix& __fastcall operator = (const Extended *Src); // ���������� ������� ������ �������
  //Matrix& __fastcall operator = (const Extended Initial); // ���������� ������� ������ �������

  //Matrix& __fastcall operator += (const Matrix& MSum);
  //Matrix __fastcall operator + (const Matrix& MS) const;
  //Matrix& __fastcall operator -= (const Matrix& MSum);
  //Matrix __fastcall operator - (const Matrix& MS) const;

  //Matrix& __fastcall operator *= (const Extended& Val);
  //Matrix& __fastcall operator /= (const Extended& Val);
  //friend Matrix __fastcall operator * (const Extended& Val, const Matrix& MMul);
  //Matrix __fastcall operator * (const Extended& Val) const;
  //Matrix __fastcall operator / (const Extended& Val) const;

  /* ����������� ������ : M1(l,m)*M2(m,n)=M3(l,n) */
  //Matrix& __fastcall operator *= (const Matrix& MMul);
  //Matrix __fastcall operator * (const Matrix& MMul) const;

  //Matrix __fastcall operator - (void) const;

  /* ��������� ������, ���� : ������� ��������� � ����� ����� ����� ��������������� �������� */
  //bool __fastcall operator == (const Matrix& MCom) const;
  //bool __fastcall operator != (const Matrix& MCom) const;
  /* ��������� ������, ���� : ��� �������� ����� ��������� ����� */
  //bool __fastcall operator == (const Extended& Val) const;
  //bool __fastcall operator != (const Extended& Val) const;

  //Extended& __fastcall operator [] (int Count); // Count = 1..(Lin*Col)

  //int __fastcall Index(int L, int C);

private:
  //int FCol;
  //int FLin;
  //Extended *M; // ������ ����� �������
  //void __fastcall Free(void); // ������������� ������, Col � Lin �� ����������
  //void __fastcall TryCreate(void); // ���� Col � Lin �� ����� ����, ������� ������ (Col*Lin) �����
  //void __fastcall SetCol(const int Val);
  //void __fastcall SetLin(const int Val);
};

}  // namespace mathlib

#endif  // MATHLIB_MATRIX_H
