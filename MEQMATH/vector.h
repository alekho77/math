/*
    Описание объекта типа "Вектор"
    08.01.2000
    (c) Хозин Алексей Ю. (Doctor Alex)
*/
//---------------------------------------------------------------------------
#ifndef vectorH
#define vectorH
class Matrix;
//---------------------------------------------------------------------------
class Vector
{
private:
    int FDim; // Размерность вектора
    Extended *V; // Массив чисел вектора
    void __fastcall Free(void); // Высвобождение памяти, Dim не изменяется
    void __fastcall TryCreate(void); // Если Dim не равен нулю, создает массив Dim чисел
    void __fastcall SetDim(const int Val);
public:
    /* Конструкторы и деструктор */
    __fastcall Vector();  // Конструктор по умолчанию
    __fastcall Vector(const int D); // Конструктор с созданием массива
    __fastcall Vector(const Vector& VSrc); // Конструктор создания копии
    __fastcall ~Vector();
    /* Общие свойства */
    __property int Dim = {read = FDim, write = SetDim}; // Размерность
    /* -= Операторы присваивания =- */
            /* Создание копии (предыдущая информация в векторе теряется) */
    Vector& __fastcall operator = (const Vector& VSrc);
            /* Инициализация вектора массивом чисел (если Src==NULL высвобождение занимаемой памяти, и Dim=0) */
    Vector& __fastcall operator = (const Extended *Src); // Необходимо зарание задать размер
    Vector& __fastcall operator = (const Extended Initial); // Необходимо зарание задать размер
    /* -= Операторы сложения и вычитания векторов =- */
            /* Размерности векторов должны совпадать */
    Vector& __fastcall operator += (const Vector& VSum);
    Vector __fastcall operator + (const Vector& VS) const;
    Vector& __fastcall operator -= (const Vector& VSum);
    Vector __fastcall operator - (const Vector& VS) const;
    /* -= Умножение и деление векторов на число =- */
    Vector& __fastcall operator *= (const Extended& Val);
    Vector& __fastcall operator /= (const Extended& Val);
    friend Vector __fastcall operator * (const Extended& Val, const Vector& VMul);
    Vector __fastcall operator * (const Extended& Val) const;
    Vector __fastcall operator / (const Extended& Val) const;
    /* -= Скалярное умножение векторов =- */
            /* Размерности векторов должны совпадать */
    Extended __fastcall operator * (const Vector& VMul) const;
    /* Умножение матрицы на вектор. Размеры : M(m,n)*V(n) = V(m) */
    friend Vector __fastcall operator * (const Matrix& M, const Vector& V);
    /* Унарные операторы */
    Vector __fastcall operator - (void) const;
    /* Операторы сравнения */
        /* Равенство только, если : Резмеры совпадают и равны между собой соответствующие элементы */
    bool __fastcall operator == (const Vector& VCom) const;
    bool __fastcall operator != (const Vector& VCom) const;
        /* Равенство только, если : Все элементы равны заданному числу */
    bool __fastcall operator == (const Extended& Val) const;
    bool __fastcall operator != (const Extended& Val) const;
    /* Индексный оператор */
    Extended& __fastcall operator [] (int Count); // Count = 1..Dim
    /* Определение модуля вектора */
    Extended __fastcall Mod(void);
};
//---------------------------------------------------------------------------
class TVectorError
{
private:
    int FCount; // Число ошибок
    int *iErr; // Массив кодов ошибок
    int __fastcall GetLastError(void);
public:
    __property int Count = { read = FCount };
    __property int LastErr = { read = GetLastError }; // Возвращает 0, если массив пуст
    __fastcall TVectorError();
    __fastcall ~TVectorError();
        /* Можно получить строку с описанием ошибки, код которой находится в массиве под номером Index */
    AnsiString __fastcall ErrStr(int Index); // Index первого элемента = 1
        /* Очишает массив с ошибками */
    void __fastcall Clear(void);
        /* Добавить ошибку */
    void __fastcall Add(int ErrCode);
};
//---------------------------------------------------------------------------
extern TVectorError *VectorError;
//---------------------------------------------------------------------------
#endif
