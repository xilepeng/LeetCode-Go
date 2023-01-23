package main

import (
	"fmt"
)

func heap_sort(Arr []int, heap_size int) {
	// heap_size := len(Arr)
	build_maxheap(Arr, heap_size)
	for i := heap_size - 1; i >= 0; i-- {
		Arr[0], Arr[i] = Arr[i], Arr[0]
		heap_size--
		max_heapify(Arr, 0, heap_size)
	}
}
func build_maxheap(A []int, heap_size int) {
	for i := heap_size >> 1; i >= 0; i-- {
		max_heapify(A, i, heap_size)
	}
}
func max_heapify(A []int, i, heap_size int) {
	l, r, largest := i<<1+1, i<<1+2, i
	for l < heap_size && A[largest] < A[l] {
		largest = l
	}
	for r < heap_size && A[largest] < A[r] {
		largest = r
	}
	if largest != i {
		A[largest], A[i] = A[i], A[largest]
		max_heapify(A, largest, heap_size)
	}
}

func main() {
	n, count := 0, 0
	fmt.Scanf("%d", &n)
	A := make([]int, n)
	for i := 0; i < n; i++ {
		fmt.Scanf("%d", &A[i])
		count++

		if count < 3 {
			fmt.Printf("-1\n")
		} else {
			heap_sort(A, count)
			j := count - 1
			for i := 1; i <= 3; i++ {
				fmt.Printf("%d ", A[j])
				j--
			}
			fmt.Printf("\n")
		}
	}
}
