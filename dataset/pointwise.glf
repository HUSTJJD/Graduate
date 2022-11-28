package require PWI_Glyph 4.18.4

pw::Application setUndoMaximumLevels 5

proc cleanGrid {} {
    set grids [pw::Grid getAll -type pw::Connector]
    if {[llength $grids]>0} {
        foreach grid $grids {$grid delete -force}
    }
}

proc cleanGeom {} {
    cleanGrid
    set dbs [pw::Database getAll]
    if {[llength $dbs]>0} {
        foreach db $dbs {$db delete -force}
    }
    pw::Application reset -keep [list Clipboard]
    pw::Display resetView -Z
    pw::Display resetRotationPoint
    pw::Application clearModified
}

proc airfoilMesh {} {

# BOUNDARY LAYER INPUTS
# -----------------------------------------------
# initDs = initial cell height
# cellGr = cell growth rate
# blDist = boundary layer distance
# numPts = number of points around airfoil
set initDs $::initds
set cellGr $::cellgr
set blDist $::bldist
set numPts $::numpts

# CONNECTOR CREATION, DIMENSIONING, AND SPACING
# -----------------------------------------------
# Get all database entities
set dbEnts [pw::Database getAll]

# Get the curve length of all db curves
foreach db $dbEnts {
    lappend crvLength [$db getLength 1.0]
}

# Find trailing edge from minimum curve length
if {[lindex $crvLength 0] < [lindex $crvLength 1]} {
    set min 0
} else {
    set min 1
}

if {[lindex $crvLength $min] < [lindex $crvLength 2]} {
    set min $min
} else {
    set min 2
}

set dbTe [lindex $dbEnts $min]

# Get upper and lower surfaces
foreach db $dbEnts {
    if {$db != $dbTe} {
        lappend upperLower $db
    }
}

# Find y values at 50 percent length of upper and lower surfaces
set y1 [lindex [[lindex $upperLower 0] getXYZ -arc 0.5] 1]
set y2 [lindex [[lindex $upperLower 1] getXYZ -arc 0.5] 1]

# Determine upper and lower surface db entities
if {$y1 < $y2} {
    set dbLower [lindex $upperLower 0]
    set dbUpper [lindex $upperLower 1]
} else {
    set dbLower [lindex $upperLower 1]
    set dbUpper [lindex $upperLower 0]
}

# Create connectors on database entities
set upperSurfCon [pw::Connector createOnDatabase $dbUpper]
set lowerSurfCon [pw::Connector createOnDatabase $dbLower]
set trailSurfCon [pw::Connector createOnDatabase $dbTe]
set cons "$upperSurfCon $lowerSurfCon $trailSurfCon"

# Calculate main airfoil connector dimensions
foreach con $cons {lappend conLen [$con getLength -arc 1]}
set upperSurfConLen [lindex $conLen 0]
set lowerSurfConLen [lindex $conLen 1]
set trailSurfConLen [lindex $conLen 2]
set conDim [expr int($numPts/2)]

# Dimension upper and lower airfoil surface connectors
$upperSurfCon setDimension $conDim
$lowerSurfCon setDimension $conDim

# Dimension trailing edge airfoil connector
set teDim [expr int($trailSurfConLen/(10*$initDs))+2]
$trailSurfCon setDimension $teDim

# Set leading and trailing edge connector spacings
set ltDs [expr 10*$initDs]

set upperSurfConDis [$upperSurfCon getDistribution 1]
set lowerSurfConDis [$lowerSurfCon getDistribution 1]
set trailSurfConDis [$trailSurfCon getDistribution 1]

$upperSurfConDis setBeginSpacing $ltDs
$upperSurfConDis setEndSpacing $ltDs
$lowerSurfConDis setBeginSpacing $ltDs
$lowerSurfConDis setEndSpacing $ltDs

set afEdge [pw::Edge createFromConnectors -single $cons]
set afDom [pw::DomainStructured create]
$afDom addEdge $afEdge

set afExtrude [pw::Application begin ExtrusionSolver $afDom]
	$afDom setExtrusionSolverAttribute NormalInitialStepSize $initDs
	$afDom setExtrusionSolverAttribute SpacingGrowthFactor $cellGr
	$afDom setExtrusionSolverAttribute NormalMarchingVector {0 0 -1}
	$afDom setExtrusionSolverAttribute NormalKinseyBarthSmoothing 3
	$afDom setExtrusionSolverAttribute NormalVolumeSmoothing 0.3
	$afDom setExtrusionSolverAttribute StopAtHeight $blDist
	$afExtrude run 1000
$afExtrude end

# set _CN(6) [pw::GridEntity getByName con-5]
# set _TMP(split_params) [list]
# lappend _TMP(split_params) [$_CN(6) getParameter -closest [pw::Application getXYZ [$_CN(6) getXYZ -parameter 0.25775646081877296]]]
# lappend _TMP(split_params) [$_CN(6) getParameter -closest [pw::Application getXYZ [$_CN(6) getXYZ -parameter 0.75696655347312769]]]
# set _TMP(PW_1) [$_CN(6) split $_TMP(split_params)]
# unset _TMP(PW_1)
# unset _TMP(split_params)

# Reset view
pw::Display resetView

}


proc Meshing {fname} {

    cleanGeom

    set imported 0

    # Load airfoil file
    pw::Database import $fname
    set imported 1

    # Mesh airfoil
    cleanGrid
    airfoilMesh


}

proc CaeExport {a} {
    pw::Application setCAESolver {ANSYS Fluent} 2
    pw::Application markUndoLevel {Set Dimension 2D}

    set _DM(1) [pw::GridEntity getByName dom-1]

    # set _CN(1) [pw::GridEntity getByName con-5-split-3]
    # set _CN(2) [pw::GridEntity getByName con-5-split-1]
    # set _CN(4) [pw::GridEntity getByName con-5-split-2]
    set _CN(4) [pw::GridEntity getByName con-5]


    set _CN(3) [pw::GridEntity getByName con-4]


    set _CN(5) [pw::GridEntity getByName con-1]
    set _CN(6) [pw::GridEntity getByName con-2]
    set _CN(7) [pw::GridEntity getByName con-3]

    pw::BoundaryCondition create
    set _TMP(PW_3) [pw::BoundaryCondition getByName bc-2]
    $_TMP(PW_3) apply [list [list $_DM(1) $_CN(4)]]
    $_TMP(PW_3) setPhysicalType -usage CAE {Pressure Far Field}
    unset _TMP(PW_3)


    # pw::BoundaryCondition create
    # set _TMP(PW_5) [pw::BoundaryCondition getByName bc-3]
    # $_TMP(PW_5) apply [list [list $_DM(1) $_CN(2)] [list $_DM(1) $_CN(1)]]
    # $_TMP(PW_5) setPhysicalType -usage CAE {Pressure Outlet}
    # unset _TMP(PW_5)

    pw::BoundaryCondition create
    set _TMP(PW_7) [pw::BoundaryCondition getByName bc-3]
    $_TMP(PW_7) apply [list [list $_DM(1) $_CN(7)] [list $_DM(1) $_CN(5)] [list $_DM(1) $_CN(6)]]
    $_TMP(PW_7) setPhysicalType -usage CAE Wall
    unset _TMP(PW_7)


    pw::VolumeCondition create
    set _TMP(PW_1) [pw::VolumeCondition getByName vc-2]
    $_TMP(PW_1) apply [list $_DM(1)]
    $_TMP(PW_1) setPhysicalType Fluid
    unset _TMP(PW_1)

    set ents [list $_DM(1)]
    set _TMP(mode_1) [pw::Application begin Modify $ents]
    $_DM(1) setOrientation JMinimum IMinimum
    $_TMP(mode_1) end
    unset _TMP(mode_1)

    set _TMP(mode_1) [pw::Application begin CaeExport [pw::Entity sort [list $_DM(1)]]]
    $_TMP(mode_1) initialize -strict -type CAE C:/Users/JJD/Desktop/Graduate/dataset/CST/cas/$a/airfoil.cas
    $_TMP(mode_1) verify
    $_TMP(mode_1) write
    $_TMP(mode_1) end
    unset _TMP(mode_1)

}


set initds 0.0005
set cellgr 1.1
set bldist 50
set numpts 400

set a 0

for { }  {$a < 1000} {incr a} {
    set fname "C:/Users/JJD/Desktop/Graduate/dataset/CST/dat/$a.dat"
    Meshing $fname
    file mkdir C:/Users/JJD/Desktop/Graduate/dataset/CST/cas/$a
    CaeExport $a 
}
