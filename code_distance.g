# C:\Program Files\GAP-v4.12.2\runtime\opt\gap-v4.12.2\pkg\qdistrnd-0.9.2\qdistrnd-0.9.2\matrices

LoadPackage("QDistRnd");

filedir := DirectoriesPackageLibrary("QDistRnd","matrices");;
# output := OutputTextFile(Filename(filedir, "distances.txt"), false);;
# SetPrintFormattingStatus(output, false);

qx_filename := "QX.mtx";;
qz_filename := "QZ.mtx";;

lisX:=ReadMTXE(Filename(filedir, qx_filename), 0);;
lisZ:=ReadMTXE(Filename(filedir, qz_filename), 0);;
GX:=lisX[3];;
GZ:=lisZ[3];;
dz := DistRandCSS(GX,GZ,1000,0,0);;
dx := DistRandCSS(GZ,GX,1000,0,0);;

Print(dx);
Print(dz);