! Choix de la precision des reels : double par defaut, simple avec -DSINGLE
module precision
  implicit none
#ifdef SINGLE
  integer, parameter :: wp = kind(1.0)
#else
  integer, parameter :: wp = kind(1.0d0)
#endif
end module precision
