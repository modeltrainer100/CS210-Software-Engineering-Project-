// 6/08/2025, Assignment 1
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int *A;
int size;
int length;
void initilzise(int size1)
{
    A = (int*)malloc(size*sizeof(int));
    size = size1;
    length = 0;
}
void display()
{
    for(int i = 0; i<length; i++)
    {
        printf("%d ", A[i]);
    }
}
void append(int x)
{
    if(length < size)
    {
        A[length] = x;
        length++;
    }
    else
    {
        printf("Array is full");
    }
}
void insert(int index, int x)
{
    if(index < 0 || length >= size)
    {
        printf("Invalid");
        return;
    }
    for(int i = length; i>index; i--)
    {
        A[i] = A[i - 1];
    }
    A[index] = x;
    length++;
}
void delete(int index)
{
    if(index < 0 || index >= length)
    {
        printf("Invalid");
        return;
    }
    for(int i = index; i < length - 1; i++)
    {
        A[i] = A[i + 1];
    }
    length--;
}
int main()
{
    initilzise(5);
    append(1);
    append(2);
    append(3);
    display();
    printf("\n");
    insert(1, 10);
    delete(2);
    display();
    return 0;
}
