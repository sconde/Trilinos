#!/bin/bash
# Purpose:

order=4

if [ $order -eq 2 ]; then
    Method_array=(SSPERK22Best SSPERK42Best SSPERK32Best SSPERK22-b1 SSPERK22-b2 SSPERK42-b1 SSPERK42-b2 SSPERK62-b1 SSPERK62-b2 ceschino24 rkf23 rkf23b fehlberg12 fehlberg12b)
    resultPath="secondOrder"
elif [ $order -eq 3 ]; then
    Method_array=(SSPERK33Best SSPERK43Best SSPERK53Best bogackishampine32 SSPERK43-b1 SSPERK43-b2)
    resultPath="thirdOrder"
elif [ $order -eq 4 ]; then
    Method_array=(merson45 fehlberg45 zonneveld43 dormandprince54 SSPERK104-b3 SSPERK104-b6 SSPERK104-b7 SSPERK104-b8)
    resultPath="fourthdOrder"
fi

inputXml=Tempus_EmbeddedPaper_VdP_numericalTests.xml
CONST_STRING="__STEPPER_NAME__"
mkdir -p $resultPath
numEl=${#Method_array[@]}
echo 'N = ' $numEl

for ((i = 0 ; i < ${numEl} ; i++)); do

    mtd_name=${Method_array[$i]}
    log_file="$resultPath/$mtd_name.log"
    final_log_file="$resultPath/$mtd_name-final.log"
    echo 'Running job (' ${i} ') :' $mtd_name
    echo 'logFile: ' $log_file

    # set the stepper name to run 
    sed "s/${CONST_STRING}/${mtd_name}/g" $inputXml > Tempus_EmbeddedPaper_VdP.xml

    # now run and output the log
    ./Tempus_EmbeddedPaper.exe > $log_file

    # grep the final work-precison info
    grep -h -A 7 "Work-Precision Info:" $log_file | tee $final_log_file

    # pre-process the log files
    #cd $resultPath
    #awk -v RS="Step       Time         dt  Abs Error  Rel Error  Order  nFail  dCompTime" -v fl="$mtd_name" -v RT='NR > 1 {print RS $0 > fl"-"(NR-1)".log"}' $log_file
    #awk -v RS="SIDAFA: integration started..." -v fl="$mtd_name" -v RT="Time integration complete." 'NR > 1 {print /RS/,/RT/ > fl"-"(NR-1)".log"}' $log_file

    #cd ..
done

