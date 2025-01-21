"""
This program allows for the execution of a reaction and subsequent sampling after reaching the stationary phase.
@author: Cristopher Tinajero
"""
import masf_library

numExperimentos = int(input("Enter the number of experiments to be performed for the software: "))
solvent = str(input("Enter the solvent: "))
longgiro = str(input("Enter the Size and Level "))

for o in range(1, numExperimentos+1):
    
    rutaArchivo = r'C:\Users\Usuario\Desktop\Experimentos Vapourtec\CTP16 CO2 CAPTURE\Parte II\Formato Excel.xlsx'

    fr1,fr2,fr3,fr4,wt1,wt2,wt3,wt4,wt5,reac,reacTemp,site,timer=masf_library.excelCapturing(rutaArchivo, o)
        
    
    if site==1: 
        if o>=1 and o<=5:
            masf_library.ReactorI_SolI(o, reac, reacTemp, wt2, fr1, fr2, fr3, fr4, 5)
        if o>6 and o<=10:
            masf_library.ReactorI_SolII(o, reac, reacTemp, wt2, fr1, fr2, fr3, fr4, 5)
        if o>10 and o<=15:
            masf_library.ReactorI_SolIII(o, reac, reacTemp, wt2, fr1, fr2, fr3, fr4, 5)
    
    if site==2: 
        if o>15 and o<=20:
            masf_library.ReactorII_SolI(o, reac, reacTemp, wt2, fr1, fr2, fr3, fr4, 5)
        if o>20 and o<=25:
            masf_library.ReactorII_SolII(o, reac, reacTemp, wt2, fr1, fr2, fr3, fr4, 5)
        if o>25 and o<=30:
            masf_library.ReactorII_SolIII(o, reac, reacTemp, wt2, fr1, fr2, fr3, fr4, 5)
    
    if site==3: 
        if o>30 and o<=35:
            masf_library.ReactorIII_SolI(o, reac, reacTemp, wt2, fr1, fr2, fr3, fr4, 5)
        if o>35 and o<=40:
            masf_library.ReactorIII_SolII(o, reac, reacTemp, wt2, fr1, fr2, fr3, fr4, 5)
        if o>40 and o<=45:
            masf_library.ReactorIII_SolIII(o, reac, reacTemp, wt2, fr1, fr2, fr3, fr4, 5)
    
    if site==4: 
        if o>45 and o<=50:
            masf_library.ReactorIV_SolI(o, reac, reacTemp, wt2, fr1, fr2, fr3, fr4, 5)
        if o>50 and o<=55:
            masf_library.ReactorIV_SolII(o, reac, reacTemp, wt2, fr1, fr2, fr3, fr4, 5)
        if o>55 and o<=60:
            masf_library.ReactorIV_SolIII(o, reac, reacTemp, wt2, fr1, fr2, fr3, fr4, 5)
   
    masf_library.sampleCollection(o, wt3, site, wt3, 5)
    masf_library.protonPlus(o,wt1,5)
 
masf_library.finishingExperiment(o, 5)

    

                
