/*
    �������� ������� ���� "�������"
    08.01.2000
    (c) ����� ������� �. (Doctor Alex)
*/
//---------------------------------------------------------------------------
#ifndef matrixH
#define matrixH

//---------------------------------------------------------------------------
class Matrix
{
private:
    int FCol;
    int FLin;
    Extended *M; // ������ ����� �������
    void __fastcall Free(void); // ������������� ������, Col � Lin �� ����������
    void __fastcall TryCreate(void); // ���� Col � Lin �� ����� ����, ������� ������ (Col*Lin) �����
    void __fastcall SetCol(const int Val);
    void __fastcall SetLin(const int Val);
public:
    /* ������������ � ���������� */
    __fastcall Matrix();  // ����������� �� ���������
    __fastcall Matrix(const int L, const int C); // ����������� � ��������� �������
    __fastcall Matrix(const Matrix& MSrc); // ����������� �������� �����
    __fastcall ~Matrix();
    /* ����� �������� */
    __property int Col = {read = FCol, write = SetCol}; // �������
    __property int Lin = {read = FLin, write = SetLin}; // ������
    /* -= ��������� ������������ =- */
            /* �������� ����� (���������� ���������� � ������� ��������) */
    Matrix& __fastcall operator = (const Matrix& MSrc);
            /* ������������� ������� �������� ����� (���� Src==NULL ������������� ���������� ������, � Lin=Col=0) */
    Matrix& __fastcall operator = (const Extended *Src); // ���������� ������� ������ �������
    Matrix& __fastcall operator = (const Extended Initial); // ���������� ������� ������ �������
    /* -= ��������� �������� � ��������� ������ =- */
            /* ������� ������ ������ ��������� */
    Matrix& __fastcall operator += (const Matrix& MSum);
    Matrix __fastcall operator + (const Matrix& MS) const;
    Matrix& __fastcall operator -= (const Matrix& MSum);
    Matrix __fastcall operator - (const Matrix& MS) const;
    /* -= ��������� � ������� ������ �� ����� =- */
    Matrix& __fastcall operator *= (const Extended& Val);
    Matrix& __fastcall operator /= (const Extended& Val);
    friend Matrix __fastcall operator * (const Extended& Val, const Matrix& MMul);
    Matrix __fastcall operator * (const Extended& Val) const;
    Matrix __fastcall operator / (const Extended& Val) const;
    /* -= ��������� ������ =- */
            /* ����������� ������ : M1(l,m)*M2(m,n)=M3(l,n) */
    Matrix& __fastcall operator *= (const Matrix& MMul);
    Matrix __fastcall operator * (const Matrix& MMul) const;
    /* ������� ��������� */
    Matrix __fastcall operator - (void) const;
    /* ��������� ��������� */
        /* ��������� ������, ���� : ������� ��������� � ����� ����� ����� ��������������� �������� */
    bool __fastcall operator == (const Matrix& MCom) const;
    bool __fastcall operator != (const Matrix& MCom) const;
        /* ��������� ������, ���� : ��� �������� ����� ��������� ����� */
    bool __fastcall operator == (const Extended& Val) const;
    bool __fastcall operator != (const Extended& Val) const;
    /* ��������� �������� */
    Extended& __fastcall operator [] (int Count); // Count = 1..(Lin*Col)
    /* ������ ��� ����������� ������ �������� �� ������ ������ � ������� */
    int __fastcall Index(int L, int C);
};
//---------------------------------------------------------------------------
class TMatrixError
{
private:
    int FCount; // ����� ������
    int *iErr; // ������ ����� ������
    int __fastcall GetLastError(void);
public:
    __property int Count = { read = FCount };
    __property int LastErr = { read = GetLastError }; // ���������� 0, ���� ������ ����
    __fastcall TMatrixError();
    __fastcall ~TMatrixError();
        /* ����� �������� ������ � ��������� ������, ��� ������� ��������� � ������� ��� ������� Index */
    AnsiString __fastcall ErrStr(int Index); // Index ������� �������� = 1
        /* ������� ������ � �������� */
    void __fastcall Clear(void);
        /* �������� ������ */
    void __fastcall Add(int ErrCode);
};
//---------------------------------------------------------------------------
extern TMatrixError *MatrixError;
//---------------------------------------------------------------------------
#endif
