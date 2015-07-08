//
//  Contains methods from SAWs.js but modified to display in a PETSc specific way

PETSc = {};

//this variable is used to organize all the data from SAWs
var sawsInfo = {};

//record if initialized the page (added appropriate divs for the diagrams and such)
var init = false;

var ind = 0;
//record what iteration we are on (remove text on second iteration)
var iteration = 0;

var title = "";
//holds the colors used in the tree drawing
var colors = ["black","red","blue","green"];



//This Function is called once (document).ready. The javascript for this was written by the PETSc code into index.html
PETSc.getAndDisplayDirectory = function(names,divEntry){
    console.log("sawsInfo");
    console.log(sawsInfo);

    if(!init) {
        $("head").append('<script src="js/parsePrefix.js"></script>');//reuse the code for parsing thru the prefix
        $("head").append('<script src="js/recordSawsData.js"></script>');//reuse the code for organizing data into sawsInfo
        $("head").append('<script src="js/utils.js"></script>');//necessary for the two js files above
        $("head").append('<script src="js/drawDiagrams.js"></script>');//contains the code to draw diagrams of the solver structure. in particular, fieldsplit and multigrid
        $("head").append('<script src="js/boxTree.js"></script>');//contains the code to draw the tree
        $("head").append('<script src="js/getCmdOptions.js"></script>');//contains the code to draw the tree
        $("body").append("<div id=\"tree\" align=\"center\"></div");
        $("body").append("<div id=\"leftDiv\" style=\"float:left;\"></div>");
        $(divEntry).appendTo("#leftDiv");
        $("body").append("<div id=\"diagram\"></div>");
        $("body").append("<div id=\"optionarea\" class=\"container\"><h3>Step 1</h3><div class=\"well\"><label>Please select your configuration set up: </label> <select id=\"mode\"><option value=\"0\">Basic</option> <option value=\"1\">All</option></select> <input type=\"button\" value=\"Go!\" id=\"go\"></div></div>")

        $("#optionarea").append("<br><br><button type=\"button\" name=\"history\" id=\"history\" class=\"btn btn-info btn-lg\">View History</button>");
        $("#optionarea").append("<div class=\"modal fade\" id=\"myModal\" role=\"dialog\"> <div class\"modal-dialog\"><div class=\"modal-content\"><div class=\"modal-header\"><button type=\"button\" class=\"close\" data-dismiss=\"modal\">&times;</button><h4 class=\"modal-title\">History</h4></div><div class=\"modal-body\"><div id=\"dataS\"></div><div id=\"Info\"></div></div><div class=\"modal-footer\"><button type=\"button\" class=\"btn btn-default\" data-dismiss=\"modal\">Close</button></div></div></div></div>");


        jQuery('#go').on('click', function(){


                     var mode = $( "#mode" ).val();

                     if (mode == 0) {
                        console.log("Basic Options");
                     } else if (mode == 1) {
                        console.log("All Options");
                        jQuery(divEntry).html("");
                        SAWs.getDirectory(null,PETSc.displayDirectory,divEntry);
                     }


        });
        //$("body").append("<div class=\"accordion\" id=\"accordion2\"><div class=\"accordion-group\"><div class=\"accordion-heading\"><a class=\"accordion-toggle\" data-toggle=\"collapse\" data-parent=\"#accordion2\" href=\"#collapseOne\">Collapsible Group Item #1</a></div><div id=\"collapseOne\" class=\"accordion-body collapse in\"><div class=\"accordion-inner\">Anim pariatur cliche...</div></div></div><div class=\"accordion-group\"><div class=\"accordion-heading\"><a class=\"accordion-toggle\" data-toggle=\"collapse\" data-parent=\"#accordion2\" href=\"#collapseTwo\">Collapsible Group Item #2</a></div><div id=\"collapseTwo\" class=\"accordion-body collapse\"><div class=\"accordion-inner\">Anim pariatur cliche...</div></div></div></div>");

        //$("body").append("<button type=\"button\" class=\"btn btn-danger\" data-toggle=\"collapse\" data-target=\"#demo\">simple collapsible</button><div id=\"demo\" class=\"collapse in\">KLJLK</div>");
 

        init = true;
    } else {
                            jQuery(divEntry).html("");

                            console.log(names);

                            SAWs.getDirectory(null,PETSc.displayDirectory,divEntry);

    }






    jQuery.getJSON('/SAWs/historyStatus',function(data){
                                         console.log("RH: " + data);

                                          if (data != 0) {
                                              //jQuery("body").append("<br><br><h2>History Active</h2>");

                                              jQuery('#history').on('click', function(){

                                               $(".modal-body #dataS").html("");

                                              $('#myModal').modal('show');
                                                                                    console.log("Main History");

                                                    for (i = 0; i < data; i++) {
                                                       $(".modal-body #dataS").append("<button onclick=\"PETSc.history("+i+")\" id=\"buttonName" + i + "\" value=\"buttonValue\">" + i + "</button>");

                                                    }


                                              })

                                          } else {

                                          }

                                       })


}


PETSc.history = function (field){

//alert(field);




 jQuery.getJSON('/SAWs/historyGet/' + field,function(data){
                                                                    //alert("Action " + i + ": " + data);
                                                                    $("#Info").html("");
                                                                    console.log("History_" + field + ": " + data);
                                                                    PETSc.displayDirectoryAsString(data.directories,"",0,"Info");
                                                                })


}

PETSc.displayDirectory = function(sub,divEntry)
{

    divEntry = "#coldiv" + ind;


    globaldirectory[divEntry] = sub;
        
    iteration ++;
    if(iteration == 2) { //remove text
        for(var i=0; i<9; i++) {
            //$("body").children().first().remove();
        }
    }

    //if($("#leftDiv").children(0).is("center")) //remove the title of the options if needed
        //$("#leftDiv").children().get(0).remove();

    console.log(sub);
    console.log(divEntry);

    if(sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories != undefined) {

        recordSawsData(sawsInfo,sub); //records data into sawsInfo

        if (sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables._title.data == "Preconditioner (PC) options" || sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables._title.data == "Krylov Method (KSP) options") {

            var SAWs_prefix = sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables["prefix"].data[0];

            if(SAWs_prefix == "(null)")
                SAWs_prefix = "";

            $("#diagram").html("");
            var data = drawDiagrams(sawsInfo,"0",parsePrefix(sawsInfo,SAWs_prefix).endtag,5,5);

            if(data != "") {
                //$("#diagram").html("<svg id=\"svgCanvas\" width='700' height='700' viewBox='0 0 2000 2000'>"+data+"</svg>");
                $("#diagram").html("<svg id=\"svgCanvas\" width=\"" + sawsInfo["0"].x_extreme/4 + "\" height=\"" + sawsInfo["0"].y_extreme/4 + "\" viewBox=\"0 0 " + sawsInfo["0"].x_extreme + " " +sawsInfo["0"].y_extreme + "\">"+data+"</svg>");
                //IMPORTANT: Viewbox determines the coordinate system for drawing. width and height will rescale the SVG to the given width and height. Things should NEVER be appended to an svg element because then we would need to use a hacky refresh which works in Chrome, but no other browsers that I know of.
            }
            calculateSizes(sawsInfo,"0");
            var svgString = getBoxTree(sawsInfo,"0",0,0);
            $("#tree").html("<svg id=\"treeCanvas\" align=\"center\" width=\"" + sawsInfo["0"].total_size.width + "\" height=\"" + sawsInfo["0"].total_size.height + "\" viewBox=\"0 0 " + sawsInfo["0"].total_size.width + " " + sawsInfo["0"].total_size.height + "\">" + svgString + "</svg>");
        }
    }


    PETSc.displayDirectoryRecursive(sub.directories,divEntry,0,"");//this method is recursive on itself and actually fills the div with text and dropdown lists

    if (sub.directories.SAWs_ROOT_DIRECTORY.variables.hasOwnProperty("__Block") && (sub.directories.SAWs_ROOT_DIRECTORY.variables.__Block.data[0] == "true")) {
        console.log("divEntry");
        console.log(divEntry);
        //jQuery(divEntry).after("<input type=\"button\" value=\"Continue\" id=\"continue\">");
        var newindex = ind - 1;
        jQuery("#coldiv" + newindex).before(" <input type=\"button\" value=\"Continue\" id=\"continue\">");
        $("#continue").after("  <input type=\"button\" value=\"Finish\" id=\"finish\">");
        jQuery('#continue').on('click', function(){
            console.log("COM");

            $("#coldiv" + newindex).collapse({
              toggle: true
            })

            $("#continue").remove();//remove self immediately
            $("#finish").remove();
            SAWs.updateDirectoryFromDisplay(divEntry);
            sub.directories.SAWs_ROOT_DIRECTORY.variables.__Block.data = ["false"];
            SAWs.postDirectory(sub);
            window.setTimeout(PETSc.getAndDisplayDirectory,1000,null,divEntry);
        });
        jQuery('#finish').on('click', function(){
            $("#finish").remove();//remove self immediately
            $("#continue").remove();
            SAWs.updateDirectoryFromDisplay(divEntry);
            sub.directories.SAWs_ROOT_DIRECTORY.variables.__Block.data = ["false"];
            sub.directories.SAWs_ROOT_DIRECTORY.directories.PETSc.directories.Options.variables.StopAsking.data = ["true"];//this is hardcoded (bad)
            SAWs.postDirectory(sub);
            window.setTimeout(PETSc.getAndDisplayDirectory,1000,null,divEntry);
        });
    } else console.log("no block property or block property is false");
}


/*
 * This function appends DOM elements to divEntry based on the JSON data in sub
 *
 */

PETSc.displayDirectoryRecursive = function(sub,divEntry,tab,fullkey)
 {
     jQuery.each(sub,function(key,value){
         fullkey = fullkey+key;//key contains things such as "PETSc" or "Options"

         console.log(fullkey + " KEY");

         if(jQuery("#"+fullkey).length == 0){
             jQuery(divEntry).append("<div id =\""+fullkey+"\"></div>")
             if (key != "SAWs_ROOT_DIRECTORY") {
                 //SAWs.tab(fullkey,tab);
 	        //jQuery("#"+fullkey).append("<b>"+ key +"<b><br>");//do not display "PETSc" nor "Options"
             }

             var descriptionSave = "";//saved description string because although the data is fetched: "description, -option, value" we wish to display it: "-option, value, description"
             var manualSave = ""; //saved manual text
             var mg_encountered = false;//record whether or not we have encountered pc=multigrid

             jQuery.each(sub[key].variables, function(vKey, vValue) {//for each variable...

                 if (vKey.substring(0,2) == "__") // __Block variable
                     return;
                 //SAWs.tab(fullkey,tab+1);
                 if (vKey[0] != '_') {//this chunk  of code adds the option name
                     if(vKey.indexOf("prefix") != -1 && sub[key].variables[vKey].data[0] == "(null)")
                         return;//do not display (null) prefix

                     if(vKey.indexOf("prefix") != -1) //prefix text
                         $("#"+fullkey).append(vKey + ":&nbsp;");
                         //var newindex = ind - 1;
                         //$("#demo" + newindex).append(vKey + ":&nbsp;");
                     else if(vKey.indexOf("ChangedMethod") == -1 && vKey.indexOf("StopAsking") == -1) { //options text
                         //options text is a link to the appropriate manual page

                         var manualDirectory = "all"; //this directory does not exist yet so links will not work for now
                         console.log(ind);
                         //$("#"+fullkey).append("<br><a href=\"http://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/" +  manualDirectory + "/" + manualSave + ".html\" title=\"" + descriptionSave + "\" id=\"data"+fullkey+vKey+j+"\">"+vKey+"&nbsp</a>");
                         var newindex = ind - 1;
                         $("#coldiv" + newindex).append("<br><a href=\"http://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/" +  manualDirectory + "/" + manualSave + ".html\" title=\"" + descriptionSave + "\" id=\"data"+fullkey+vKey+j+"\">"+vKey+"&nbsp</a>");
                     }
                 }

                 for(j=0;j<sub[key].variables[vKey].data.length;j++){//vKey tells us a lot of information on what the data is. data.length is 1 most of the time. when it is more than 1, that results in 2 input boxes right next to each other

                     if(vKey.indexOf("man") != -1) {//do not display manual, but record the text
                         manualSave = sub[key].variables[vKey].data[j];
                         continue;
                     }

                     if(vKey.indexOf("title") != -1) {//display title in center

                         if (title != sub[key].variables[vKey].data[j]) {
                             console.log("<---------------------------------------------->");
                             if (ind == 0) {
                               $("body").append("<br><div id=\"buttonarea" + ind + "\" class=\"container\"><h3>Step 2</h3><button type=\"button\" class=\"btn btn-info\" data-toggle=\"collapse\" data-target=\"#coldiv" + ind + "\">"+sub[key].variables[vKey].data[j]+"</button><div id=\"coldiv" + ind + "\" class=\"collapse in\"></div></div>");
                             } else {
                               $("body").append("<br><div id=\"buttonarea" + ind + "\" class=\"container\"><button type=\"button\" class=\"btn btn-info\" data-toggle=\"collapse\" data-target=\"#coldiv" + ind + "\">"+sub[key].variables[vKey].data[j]+"</button><div id=\"coldiv" + ind + "\" class=\"collapse in\"></div></div>");
                             }
                             ind = ind + 1;
                             title = sub[key].variables[vKey].data[j];
                         } else {
                             console.log("<---------------------------------------------->");
                             var newindex = ind - 1;
                             $("#buttonarea" + newindex).remove();
                             $("body").append("<div id=\"buttonarea" + ind + "\" class=\"container\"><button type=\"button\" class=\"btn btn-info\" data-toggle=\"collapse\" data-target=\"#coldiv" + ind + "\">"+sub[key].variables[vKey].data[j]+"</button><div id=\"coldiv" + ind + "\" class=\"collapse in\"></div></div>");
                             ind = ind + 1;
                             title = sub[key].variables[vKey].data[j];
                         }


                         continue;
                     }

                     if(sub[key].variables[vKey].alternatives.length == 0) {//case where there are no alternatives
                         if(sub[key].variables[vKey].dtype == "SAWs_BOOLEAN") {

                             console.log("A: " + fullkey)
                             var newindex = ind - 1;
                             $("#coldiv" + newindex).append("<select id=\"data"+fullkey+vKey+j+"\">");//make the boolean dropdown list.
                             console.log("Test_Check:" + fullkey+vKey+j);
                             $("#data"+fullkey+vKey+j).append("<option value=\"true\">True</option> <option value=\"false\">False</option>");
                             if(vKey == "ChangedMethod" || vKey == "StopAsking") {//do not show changedmethod nor stopasking to user
                                 $("#data"+fullkey+vKey+j).attr("hidden",true);
                             }

                         } else {
                             if(sub[key].variables[vKey].mtype != "SAWs_WRITE") {

                                 descriptionSave = sub[key].variables[vKey].data[j];

                                 if(vKey.indexOf("prefix") != -1)  {

                                    var newindex = ind - 1;
                                    $("#coldiv" + newindex).append("<a style=\"font-family: Courier\" size=\""+(sub[key].variables[vKey].data[j].toString().length+1)+"\" id=\"data"+fullkey+vKey+j+"\">"+sub[key].variables[vKey].data[j]+"</a><br>");

                                 }
                             }
                             else {//can be changed (append dropdown list)
                                 var newindex = ind - 1;

                                 $("#coldiv" + newindex).append("<input type=\"text\" style=\"font-family: Courier\" size=\""+(sub[key].variables[vKey].data[j].toString().length+1)+"\" id=\"data"+fullkey+vKey+j+"\" name=\"data\" \\>");

                             }
                             jQuery("#data"+fullkey+vKey+j).keyup(function(obj) {
                                 console.log( "Key up called "+key+vKey );
                                 sub[key].variables[vKey].selected = 1;
                                 $("#data"+fullkey+"ChangedMethod0").find("option[value='true']").attr("selected","selected");//set changed to true automatically
                             });
                         }
                         jQuery("#data"+fullkey+vKey+j).val(sub[key].variables[vKey].data[j]);//set val from server
                         if(vKey != "ChangedMethod") {
                             jQuery("#data"+fullkey+vKey+j).change(function(obj) {
                                 sub[key].variables[vKey].selected = 1;
                                 $("#data"+fullkey+"ChangedMethod0").find("option[value='true']").attr("selected","selected");//set changed to true automatically
                             });
                         }
                     } else {//case where there are alternatives
                         /*
                         jQuery("#"+fullkey).append("<select id=\"data"+fullkey+vKey+j+"\">");
                         jQuery("#data"+fullkey+vKey+j).append("<option value=\""+sub[key].variables[vKey].data[j]+"\">"+sub[key].variables[vKey].data[j]+"</option>");
                         for(var l=0;l<sub[key].variables[vKey].alternatives.length;l++) {
                             jQuery("#data"+fullkey+vKey+j).append("<option value=\""+sub[key].variables[vKey].alternatives[l]+"\">"+sub[key].variables[vKey].alternatives[l]+"</option>");
                         }
                         jQuery("#"+fullkey).append("</select>");

                         jQuery("#data"+fullkey+vKey+j).change(function(obj) {
                             sub[key].variables[vKey].selected = 1;
                             $("#data"+fullkey+"ChangedMethod0").find("option[value='true']").attr("selected","selected");//set changed to true automatically
                             var id = "data"+fullkey+vKey+j;
                             if(id.indexOf("type") != -1) {//if some type variable changed, then act as if continue button was clicked
                                 $("#continue").trigger("click");
                             }
                         });
                         */
                         var newindex = ind - 1;
                         jQuery("#coldiv" + newindex).append("<select id=\"data"+fullkey+vKey+j+"\">");

                                                 jQuery("#data"+fullkey+vKey+j).append("<option value=\""+sub[key].variables[vKey].data[j]+"\">"+sub[key].variables[vKey].data[j]+"</option>");
                                                 for(var l=0;l<sub[key].variables[vKey].alternatives.length;l++) {
                                                     jQuery("#data"+fullkey+vKey+j).append("<option value=\""+sub[key].variables[vKey].alternatives[l]+"\">"+sub[key].variables[vKey].alternatives[l]+"</option>");
                                                 }
                                                 jQuery("#coldiv" + newindex).append("</select>");

                                                 jQuery("#data"+fullkey+vKey+j).change(function(obj) {
                                                     sub[key].variables[vKey].selected = 1;
                                                     $("#data"+fullkey+"ChangedMethod0").find("option[value='true']").attr("selected","selected");//set changed to true automatically
                                                     var id = "data"+fullkey+vKey+j;
                                                     if(id.indexOf("type") != -1) {//if some type variable changed, then act as if continue button was clicked
                                                         $("#continue").trigger("click");
                                                     }
                        });
                     }
                 }
             });

             if(typeof sub[key].directories != 'undefined'){
                 PETSc.displayDirectoryRecursive(sub[key].directories,divEntry,tab+1,fullkey);
              }
         }
     });
 }


PETSc.displayDirectoryAsString = function(sub,divEntry,tab,fullkey)
 {

     jQuery.each(sub,function(key,value){
              fullkey = fullkey+key;
              console.log(fullkey + " KEY");
              if(jQuery("#"+fullkey).length == 0){
                  jQuery(divEntry).append("<div id =\""+fullkey+"\"></div>")
                  if (key != "SAWs_ROOT_DIRECTORY") {

                  }

                  var descriptionSave = ""; var manualSave = ""; var mg_encountered = false;
                  jQuery.each(sub[key].variables, function(vKey, vValue) {

                      if (vKey.substring(0,2) == "__") // __Block variable
                          return;
                      if (vKey[0] != '_') {
                          if(vKey.indexOf("prefix") != -1 && sub[key].variables[vKey].data[0] == "(null)")
                              return;

                          if(vKey.indexOf("prefix") != -1)
                              $("#"+fullkey).append(vKey + ":&nbsp;");

                          else if(vKey.indexOf("ChangedMethod") == -1 && vKey.indexOf("StopAsking") == -1) {

                              var manualDirectory = "all"; var newindex = ind - 1;
                              $("#coldiv" + newindex).append("<br><a href=\"http://www.mcs.anl.gov/petsc/petsc-dev/docs/manualpages/" +  manualDirectory + "/" + manualSave + ".html\" title=\"" + descriptionSave + "\" id=\"data"+fullkey+vKey+j+"\">"+vKey+"&nbsp</a>");
                          }
                      }

                      for(j=0;j<sub[key].variables[vKey].data.length;j++){
                          if(vKey.indexOf("man") != -1) {
                              manualSave = sub[key].variables[vKey].data[j];
                              continue;
                          }

                          if(vKey.indexOf("title") != -1) {

                              if (title != sub[key].variables[vKey].data[j]) {

                                  $("#Info").append("<br><div id=\"buttonarea" + ind + "\" class=\"container\"><button type=\"button\" class=\"btn btn-info\" data-toggle=\"collapse\" data-target=\"#coldiv" + ind + "\">"+sub[key].variables[vKey].data[j]+"</button><div id=\"coldiv" + ind + "\" class=\"collapse in\"></div></div>");

                                  ind = ind + 1;

                              } else {
                                  var newindex = ind - 1;
                                  $("#buttonarea" + newindex).remove();
                                  $("#Info").append("<div id=\"buttonarea" + ind + "\" class=\"container\"><button type=\"button\" class=\"btn btn-info\" data-toggle=\"collapse\" data-target=\"#coldiv" + ind + "\">"+sub[key].variables[vKey].data[j]+"</button><div id=\"coldiv" + ind + "\" class=\"collapse in\"></div></div>");
                                  ind = ind + 1;
                              }


                              continue;
                          }

                          if(sub[key].variables[vKey].alternatives.length == 0) {//case where there are no alternatives
                              if(sub[key].variables[vKey].dtype == "SAWs_BOOLEAN") {

                                  console.log("A: " + fullkey)
                                  var newindex = ind - 1;
                                  $("#coldiv" + newindex).append("<select id=\"data"+fullkey+vKey+j+"\">");//make the boolean dropdown list.
                                  console.log("Test_Check:" + fullkey+vKey+j);
                                  $("#data"+fullkey+vKey+j).append("<option value=\"true\">True</option> <option value=\"false\">False</option>");
                                  if(vKey == "ChangedMethod" || vKey == "StopAsking") {//do not show changedmethod nor stopasking to user
                                      $("#data"+fullkey+vKey+j).attr("hidden",true);
                                  }

                              } else {
                                  if(sub[key].variables[vKey].mtype != "SAWs_WRITE") {

                                      descriptionSave = sub[key].variables[vKey].data[j];

                                      if(vKey.indexOf("prefix") != -1)  {

                                         var newindex = ind - 1;
                                         $("#coldiv" + newindex).append("<a style=\"font-family: Courier\" size=\""+(sub[key].variables[vKey].data[j].toString().length+1)+"\" id=\"data"+fullkey+vKey+j+"\">"+sub[key].variables[vKey].data[j]+"</a><br>");

                                      }
                                  }
                                  else {//can be changed (append dropdown list)
                                      var newindex = ind - 1;

                                      $("#coldiv" + newindex).append("<input type=\"text\" style=\"font-family: Courier\" size=\""+(sub[key].variables[vKey].data[j].toString().length+1)+"\" id=\"data"+fullkey+vKey+j+"\" name=\"data\" \\>");

                                  }

                              }
                              jQuery("#data"+fullkey+vKey+j).val(sub[key].variables[vKey].data[j]);//set val from server
                              if(vKey != "ChangedMethod") {

                              }
                          } else {

                              var newindex = ind - 1;
                              jQuery("#coldiv" + newindex).append("<select id=\"data"+fullkey+vKey+j+"\">");

                                                      jQuery("#data"+fullkey+vKey+j).append("<option value=\""+sub[key].variables[vKey].data[j]+"\">"+sub[key].variables[vKey].data[j]+"</option>");
                                                      for(var l=0;l<sub[key].variables[vKey].alternatives.length;l++) {
                                                          jQuery("#data"+fullkey+vKey+j).append("<option value=\""+sub[key].variables[vKey].alternatives[l]+"\">"+sub[key].variables[vKey].alternatives[l]+"</option>");
                                                      }
                                                      jQuery("#coldiv" + newindex).append("</select>");



                          }
                      }
                  });

                  if(typeof sub[key].directories != 'undefined'){
                      PETSc.displayDirectoryAsString(sub[key].directories,divEntry,tab+1,fullkey);
                   }
              }
          });
      }
