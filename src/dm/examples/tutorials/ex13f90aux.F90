module ex13f90aux 
  implicit none
contains
  !
  ! A subroutine which returns the boundary conditions. Probably superfluous. 
  !
  subroutine get_boundary_cond(b_x,b_y,b_z)
#include <finclude/petscsysdef.h>
#include <finclude/petscdmdef.h>
#include <finclude/petscdmda.h>
#include <finclude/petscdmda.h90>
    DMDABoundaryType,intent(inout) :: b_x,b_y,b_z
    
    ! Here you may do your own BC stuff
    b_x = DMDA_BOUNDARY_GHOSTED
    b_y = DMDA_BOUNDARY_GHOSTED
    b_z = DMDA_BOUNDARY_GHOSTED
   
  end subroutine get_boundary_cond
  !
  ! A function which returns the RHS of the equation we are solving
  !
  function dfdt_vdp(t,dt,ib1,ibn,jb1,jbn,kb1,kbn,imax,jmax,kmax,n,f)
    !
    ! Right-hand side for the van der Pol oscillator.  Very simple system of two
    ! ODEs.  See Iserles, eq (5.2).
    !
    double precision, intent(in) :: t,dt
    integer, intent(in) :: ib1,ibn,jb1,jbn,kb1,kbn,imax,jmax,kmax,n
    double precision, dimension(n,ib1:ibn,jb1:jbn,kb1:kbn), intent(inout) :: f
    double precision, dimension(n,imax,jmax,kmax) :: dfdt_vdp
    double precision, parameter :: mu=1.0
    !
    dfdt_vdp(1,:,:,:) = f(2,1,1,1)
    dfdt_vdp(2,:,:,:) = mu*(1.0 - f(1,1,1,1)**2)*f(2,1,1,1) - f(1,1,1,1)
  end function dfdt_vdp
  !
  ! The standard Forward Euler time-stepping method.
  !
  recursive subroutine forw_euler(t,dt,ib1,ibn,jb1,jbn,kb1,kbn,&
                                     imax,jmax,kmax,neq,y,dfdt)
    implicit none
    double precision, intent(in) :: t,dt
    integer, intent(in) :: ib1,ibn,jb1,jbn,kb1,kbn,imax,jmax,kmax,neq
    double precision, dimension(neq,ib1:ibn,jb1:jbn,kb1:kbn), intent(inout) :: y
    !
    ! Define the right-hand side function
    !
    interface
      function dfdt(t,dt,ib1,ibn,jb1,jbn,kb1,kbn,imax,jmax,kmax,n,f)
        double precision, intent(in) :: t,dt
        integer, intent(in) :: ib1,ibn,jb1,jbn,kb1,kbn,imax,jmax,kmax,n
        double precision, dimension(n,ib1:ibn,jb1:jbn,kb1:kbn), intent(inout) :: f
        double precision, dimension(n,imax,jmax,kmax) :: dfdt
      end function dfdt
    end interface
    !--------------------------------------------------------------------------
    !
    y(:,1:imax,1:jmax,1:kmax) = y(:,1:imax,1:jmax,1:kmax) &
                              + dt*dfdt(t,dt,ib1,ibn,jb1,jbn,kb1,kbn,imax,jmax,kmax,neq,y)
  end subroutine forw_euler
  !
  ! The following 4 subroutines handle the mapping of coordinates. I'll explain
  ! this in detail:
  !    PETSc gives you local arrays which are indexed using the global indices.
  ! This is probably handy in some cases, but when you are re-writing an
  ! existing serial code and want to use DMDAs, you have tons of loops going
  ! from 1 to imax etc. that you don't want to change. 
  !    These subroutines re-map the arrays so that all the local arrays go from
  ! 1 to the (local) imax. 
  !
  subroutine petsc_to_local(da,vec,array,f,dof,stw)
#include <finclude/petscsysdef.h>
#include <finclude/petscvecdef.h>
#include <finclude/petscdmdef.h>
#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscdmda.h>
#include <finclude/petscvec.h90>
#include <finclude/petscdmda.h90>
    DM                                                    :: da
    Vec,intent(in)                                        :: vec
    PetscScalar, pointer, intent(inout)                   :: array(:,:,:,:)
    integer,intent(in)                                    :: dof,stw
    double precision,intent(inout),dimension(:,1-stw:,1-stw:,1-stw:)  :: f
    PetscErrorCode                                        :: ierr
    !
    call DMDAVecGetArrayF90(da,vec,array,ierr)
    call transform_petsc_us(array,f,stw)
  end subroutine petsc_to_local
  subroutine transform_petsc_us(array,f,stw)
    !Note: this assumed shape-array is what does the "coordinate transformation"
    integer,intent(in)                                   :: stw
    double precision, intent(in), dimension(:,1-stw:,1-stw:,1-stw:)  :: array
    double precision,intent(inout),dimension(:,1-stw:,1-stw:,1-stw:) :: f
    f(:,:,:,:) = array(:,:,:,:)
  end subroutine transform_petsc_us
  subroutine local_to_petsc(da,vec,array,f,dof,stw)
#include <finclude/petscsysdef.h>
#include <finclude/petscvecdef.h>
#include <finclude/petscdmdef.h>
#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscdmda.h>
#include <finclude/petscvec.h90>
#include <finclude/petscdmda.h90>
    DM                                                    :: da
    Vec,intent(inout)                                     :: vec
    PetscScalar, pointer, intent(inout)                   :: array(:,:,:,:)
    integer,intent(in)                                    :: dof,stw
    double precision,intent(inout),dimension(:,1-stw:,1-stw:,1-stw:)  :: f
    PetscErrorCode                                        :: ierr
    call transform_us_petsc(array,f,stw)
    call DMDAVecRestoreArrayF90(da,vec,array,ierr)
  end subroutine local_to_petsc
  subroutine transform_us_petsc(array,f,stw)
    !Note: this assumed shape-array is what does the "coordinate transformation"
    integer,intent(in)                                     :: stw
    double precision, intent(inout), dimension(:,1-stw:,1-stw:,1-stw:) :: array
    double precision,intent(in),dimension(:,1-stw:,1-stw:,1-stw:)      :: f
    array(:,:,:,:) = f(:,:,:,:)
  end subroutine transform_us_petsc
end module ex13f90aux
