// Definir kernel
__global__ void adicionarVetores(float* A, float* B, float* C)
{
    int i = threadIdx.x; // Indice definido com base na tarefa
    C[i] = A[i] + B[i];
}

int main()
{
    // ...
    // Invocar kernel com N tarefas
    adicionarVetores<<<1, N>>>(A, B, C);
    // ...

    return 0;
}
