The output array is computed using the stencil operation, where each cell's value is the sum of the cell's value in the input array and its immediate neighbours (left, right, up, down).

For the interior elements (not on the borders), the computation is as follows: the value is the sum of the cell's value and its four neighbours (left, right, up, down).

For the border elements (e.g., the elements at index [1, 1], [1, 2], [1, 3], ..., [8, 8]), they only have three or fewer neighbours due to the edge of the grid. Therefore, their value is the sum of the cell's value and its immediate neighbours, considering the edges.

The interior elements in the output array are computed correctly according to the stencil operation.

However, the border elements are copied directly from the input array to the output array because the stencil operation cannot be fully applied to them since they do not have all their neighbours.

So, the values in the border regions of the output array remain unchanged from the input array, while the interior regions are modified based on the stencil operation.
