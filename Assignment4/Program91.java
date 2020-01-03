import java.util.Scanner;
public class PredictGPA {
    public static void main(String [] arg){
        Scanner scnr = new Scanner(System.in);

        double numHour1;
        double numHour2;
        double numHour3;
        double numHour4;
        String grade1;
        int gradeValue1 = 0;
        double totPtsClass1;
        String grade2;
        int gradeValue2 = 0;
        double totPtsClass2;
        String grade3;
        int gradeValue3 = 0;
        double totPtsClass3;
        String grade4;
        int gradeValue4 = 0;
        double totPtsClass4;
        double totPts;
        double totNumHour;
        double GPA;



        System.out.println("Course 1: Number of hours? ");
        numHour1 = scnr.nextDouble();
        System.out.println("Course 1: Expected grade? ");
        grade1 = scnr.next();

        if(grade1.equals ("A")) {
            gradeValue1 = 4;
        }
        else if(grade1.equals ("B")) {
            gradeValue1 = 3;
        }
        else if(grade1.equals ("C")){
            gradeValue1 = 2;
        }
        else if(grade1.equals ("D")){
            gradeValue1 = 1;
        }
        totPtsClass1 = gradeValue1 * numHour1;


        System.out.println("Course 2: Number of hours? ");
        numHour2 = scnr.nextDouble();
        System.out.println("Course 2: Expected grade? ");
        grade2 = scnr.next();

        if(grade2.equals ("A")) {
            gradeValue2 = 4;
        }
        else if(grade2.equals ("B")) {
            gradeValue2 = 3;
        }
        else if(grade2.equals ("C")){
            gradeValue2 = 2;
        }
        else if(grade2.equals ("D")){
            gradeValue2 = 1;
        }
        totPtsClass2 = gradeValue2 * numHour2;


        System.out.println("Course 3: Number of hours? ");
        numHour3 = scnr.nextDouble();
        System.out.println("Course 3: Expected grade? ");
        grade3 = scnr.next();

        if(grade3.equals ("A")) {
            gradeValue3 = 4;
        }
        else if(grade3.equals ("B")) {
            gradeValue3 = 3;
        }
        else if(grade3.equals ("C")){
            gradeValue3 = 2;
        }
        else if(grade3.equals ("D")){
            gradeValue3 = 1;
        }
        totPtsClass3 = gradeValue3 * numHour3;


        System.out.println("Course 4: Number of hours? ");
        numHour4 = scnr.nextDouble();
        System.out.println("Course 4: Expected grade? ");
        grade4 = scnr.next();

        if(grade4.equals ("A")) {
            gradeValue4 = 4;
        }
        else if(grade4.equals ("B")) {
            gradeValue4 = 3;
        }
        else if(grade4.equals ("C")){
            gradeValue4 = 2;
        }
        else if(grade4.equals ("D")){
            gradeValue4 = 1;
        }
        totPtsClass4 = gradeValue4 * numHour4;


        totPts = totPtsClass1 + totPtsClass2 + totPtsClass3 + totPtsClass4;
        totNumHour = numHour1 + numHour2 + numHour3 + numHour4;
        GPA = totPts / totNumHour;


        System.out.printf("Your semester GPA would be: %.2f", GPA);






        }


}
