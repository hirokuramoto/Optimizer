subroutine test(n, n_param, b, A, gram_matrix)
  implicit none
  integer i, j
  integer(4), intent(in) :: n
  integer(4), intent(in) :: n_param
  real(8),    intent(in) :: A(0 : n - 1, 0 : n_param - 1)
  real(8),    intent(in) :: b
  real(8),    intent(inout):: gram_matrix(0 : n - 1, 0 : n - 1)

  do i = 0 , n - 1
    do j = i + 1, n - 1
      gram_matrix(j, i) = exp(-b * dot_product((A(i, :)- A(j, :)), (A(i, :)- A(j, :))))
      gram_matrix(i, i) = 1
      gram_matrix(i, j) = gram_matrix(j, i)
    end do
  end do

  !write(*, "(a)") "** write from fortran"
  !write(*, "(i8)")n
  !write(*, "(1p3e12.4)")b

  !write(*, "(1p3e12.4)")gram_matrix(0, 0), gram_matrix(0, 1), gram_matrix(0, 2)
  !write(*, "(1p3e12.4)")gram_matrix(1, 0), gram_matrix(1, 1), gram_matrix(1, 2)
  !write(*, "(1p3e12.4)")gram_matrix(2, 0), gram_matrix(2, 1), gram_matrix(2, 2)

  !write(*, "(1p3e12.4)")A(0, 0), A(0, 1), A(0, 2)
  !write(*, "(1p3e12.4)")A(1, 0), A(1, 1), A(1, 2)
  !write(*, "(1p3e12.4)")A(2, 0), A(2, 1), A(2, 2)

end subroutine test
