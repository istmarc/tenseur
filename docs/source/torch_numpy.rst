Tenseur for torch and numpy users
=================================

.. list-table:: Types
   :widths: 50 25 25
   :header-rows: 1

   * - Tenseur
     - Torch
     - Numpy
   * - ten::tensor<T, Rank>
     - torch.Tensor
     - np.ndarray
   * - ten::matrix<T>
     - torch.Tensor
     - np.ndarray
   * - ten::vector<T>
     - torch.Tensor
     - np.ndarray
   * - ten::stensor<Dims...>
     - x
     - x
   * - ten::smatrix<Rows, Cols>
     - x
     - x
   * - ten::svector<Size>
     - x
     - x
   * - float
     - torch.float32; torch.float
     - np.float32
   * - double
     - torch.float64; torch.double
     - np.float64
   * - x
     - torch.float16; torch.half
     - np.float16
   * - x
     - torch.int8
     - np.int8
   * - x
     - torch.uint8
     - np.uint8
   * - x
     - torch.int16; torch.short
     - np.int16
   * - int32_t
     - torch.int32; torch.int
     - np.int32
   * - uint32_t
     - x
     - x
   * - int64_t
     - torch.int64; torch.long
     - np.int64
   * - uint64_t
     - x
     - x

