/*
    �������� ������� ���� "������"
    08.01.2000
    (c) ����� ������� �. (Doctor Alex)
*/
//---------------------------------------------------------------------------
#ifndef vectorH
#define vectorH
class Matrix;
//---------------------------------------------------------------------------
class Vector
{
private:
    int FDim; // ����������� �������
    Extended *V; // ������ ����� �������
    void __fastcall Free(void); // ������������� ������, Dim �� ����������
    void __fastcall TryCreate(void); // ���� Dim �� ����� ����, ������� ������ Dim �����
    void __fastcall SetDim(const int Val);
public:
    /* ������������ � ���������� */
    __fastcall Vector();  // ����������� �� ���������
    __fastcall Vector(const int D); // ����������� � ��������� �������
    __fastcall Vector(const Vector& VSrc); // ����������� �������� �����
    __fastcall ~Vector();
    /* ����� �������� */
    __property int Dim = {read = FDim, write = SetDim}; // �����������
    /* -= ��������� ������������ =- */
            /* �������� ����� (���������� ���������� � ������� ��������) */
    Vector& __fastcall operator = (const Vector& VSrc);
            /* ������������� ������� �������� ����� (���� Src==NULL ������������� ���������� ������, � Dim=0) */
    Vector& __fastcall operator = (const Extended *Src); // ���������� ������� ������ ������
    Vector& __fastcall operator = (const Extended Initial); // ���������� ������� ������ ������
    /* -= ��������� �������� � ��������� �������� =- */
            /* ����������� �������� ������ ��������� */
    Vector& __fastcall operator += (const Vector& VSum);
    Vector __fastcall operator + (const Vector& VS) const;
    Vector& __fastcall operator -= (const Vector& VSum);
    Vector __fastcall operator - (const Vector& VS) const;
    /* -= ��������� � ������� �������� �� ����� =- */
    Vector& __fastcall operator *= (const Extended& Val);
    Vector& __fastcall operator /= (const Extended& Val);
    friend Vector __fastcall operator * (const Extended& Val, const Vector& VMul);
    Vector __fastcall operator * (const Extended& Val) const;
    Vector __fastcall operator / (const Extended& Val) const;
    /* -= ��������� ��������� �������� =- */
            /* ����������� �������� ������ ��������� */
    Extended __fastcall operator * (const Vector& VMul) const;
    /* ��������� ������� �� ������. ������� : M(m,n)*V(n) = V(m) */
    friend Vector __fastcall operator * (const Matrix& M, const Vector& V);
    /* ������� ��������� */
    Vector __fastcall operator - (void) const;
    /* ��������� ��������� */
        /* ��������� ������, ���� : ������� ��������� � ����� ����� ����� ��������������� �������� */
    bool __fastcall operator == (const Vector& VCom) const;
    bool __fastcall operator != (const Vector& VCom) const;
        /* ��������� ������, ���� : ��� �������� ����� ��������� ����� */
    bool __fastcall operator == (const Extended& Val) const;
    bool __fastcall operator != (const Extended& Val) const;
    /* ��������� �������� */
    Extended& __fastcall operator [] (int Count); // Count = 1..Dim
    /* ����������� ������ ������� */
    Extended __fastcall Mod(void);
};
//---------------------------------------------------------------------------
class TVectorError
{
private:
    int FCount; // ����� ������
    int *iErr; // ������ ����� ������
    int __fastcall GetLastError(void);
public:
    __property int Count = { read = FCount };
    __property int LastErr = { read = GetLastError }; // ���������� 0, ���� ������ ����
    __fastcall TVectorError();
    __fastcall ~TVectorError();
        /* ����� �������� ������ � ��������� ������, ��� ������� ��������� � ������� ��� ������� Index */
    AnsiString __fastcall ErrStr(int Index); // Index ������� �������� = 1
        /* ������� ������ � �������� */
    void __fastcall Clear(void);
        /* �������� ������ */
    void __fastcall Add(int ErrCode);
};
//---------------------------------------------------------------------------
extern TVectorError *VectorError;
//---------------------------------------------------------------------------
#endif
