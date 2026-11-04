! Mesure de performance : nombre d'iterations fixe, temps par iteration.
! Usage : ./performance n iterations
program performance
  use precision
  use solveur
  implicit none
  real(wp), allocatable :: u(:,:), f(:,:)
  real(wp) :: ecart
  integer :: n, maxit, iters
  integer(8) :: t0, t1, freq
  character(len=32) :: arg
  real(8) :: ms

  call get_command_argument(1, arg); read(arg,*) n
  call get_command_argument(2, arg); read(arg,*) maxit
  allocate(u(n,n), f(n,n))
  call initialiser(n, u, f)
  call system_clock(t0, freq)
  call jacobi(n, u, f, -1.0_wp, maxit, iters, ecart)
  call system_clock(t1)
  ms = 1000.0d0 * real(t1 - t0, 8) / real(freq, 8) / real(iters, 8)
  ! sortie CSV : n, iterations, ms par iteration, Mpoints/s, octets par reel
  print '(i0,",",i0,",",f10.4,",",f10.2,",",i0)', n, iters, ms, &
        real(n-2,8)**2 / (ms*1.0d-3) / 1.0d6, storage_size(u(1,1))/8
end program performance
