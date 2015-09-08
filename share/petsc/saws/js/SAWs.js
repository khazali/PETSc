/*
 John O'Sullivan
 Copyright (c) 2013 UChicago Argonne, LLC

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*/

/*
   Provides a simple library of routines for getting and posting SAWs directories and values to the server and 
   displaying them in a browser
*/

/*
   Userful hints about jQuery and javascript syntax
   ------------------------------------------------

    jQuery.each(dictionary,function(key,value)) --- runs through the dictionary calling the function on each key,value pair
         - the function is called sequentially and synchronously once for each key, value pair
         - the function has access to the variables in the outer function (it is a "closure")
         - the outer function does not continue until all the key,value pairs are processed

    jQuery.getJSON(url,function(data)) ---- gets the data from the URL on the server and calls the function with that data 
         - the function is called asynchronously once the data arrives
         - the function has access to the variables in the outer function
         - the outer function continues possibly before the inner function is called or completed

    jQuery("#variablesInfo").  --- accesses the DOM object on the webpage name #variablesInto, i.e. <div id="variablesInfo" 

*/

/*name (or names), as referred to below, is a specific data structure that allows for fine grain
  choice of which directories (and variables) to get from the server. 
  names = {'dirName1':['varName1','varName2'],'dirName2':['varName1','varName2'],'dirName3':[],'dirName4:['varName4']}
  when the array [] is empty it requests all the variables
*/

/*
   SAWs functions are all collect in the SAWs namespace
*/
var SAWs = {}
var ind = 0;
var isMap = true;
var skipWrite = false;
var map;
var directionsDisplay;
/*
  SAWs.getDirectory grabs the directories and variables listed in names  from the server

  For each entry in names (or once if names is null) the callback is called with the received information for that entry

*/

var ws;

ws = new WebSocket("ws://localhost:8088/echo");

ws.onopen = function() {

console.log("Connection Created!");

};

ws.onerror = function() {
                        console.log("Error webosocket closed");
                    };

                    ws.onmessage = function(event) {
                        console.log("Data: " + event.data);
                    };

SAWs.getDirectory = function(names,callback,callbackdata) {

  /*If names is null, get all*/
  if(names == null){

    jQuery.getJSON('/SAWs/*',function(data){
                                       if(typeof(callback) == typeof(Function)) callback(data,callbackdata)
                                       console.log(data);
                                    })

  } else {
    jQuery.getJSON('/SAWs/' + names,function(data){
                                       if(typeof(callback) == typeof(Function)) callback(data,callbackdata)
                                       console.log(data);
                                    })

 //   jQuery.each(names,function(key,value){
   //                     var directory
     //                   directory = key + "/"
       //                 if(names[key].length == 0){
         //                  jQuery.getJSON('/SAWs/' + directory,function(data){
           //                                                      if(typeof(callback) == typeof(Function)) callback(data,callbackdata)
             //                                                  })
               //          } else {
                 //          for(var j = 0;j<names[key].length;j++){
                   //          jQuery.getJSON('/SAWs/' + directory + names[key][j],function(data){
                     //                                                              if(typeof(callback) == typeof(Function)) callback(data,callbackdata)
                       //                                                         })
                   //        }
                 //        }
                   //   })
  }


};


/*  
   SAWs.getAndDisplayDirectory(divEntry,names) - Gets the lastest values from the server and calls SAWs.displayDirectory() to display them
*/
SAWs.getAndDisplayDirectory = function(names,divEntry){
  //$("head").append('<link rel="stylesheet" type="text/css" href="css/bootstrap.css">');//reuse the code for parsing thru the prefix



  jQuery(divEntry).html("")
  SAWs.getDirectory(names,SAWs.displayDirectory,divEntry)


   jQuery.getJSON('/SAWs/historyStatus',function(data){

                                      if (data != 0) {

                                          $("#variablesInfo").before("<div class=\"container\"><button type=\"button\" name=\"history\" id=\"history\" class=\"btn btn-info btn-lg\">View History</button></div><br><br>");
                                          $("#variablesInfo").append("<div class=\"modal fade\" id=\"myModal\" role=\"dialog\"> <div class\"modal-dialog\"><div class=\"modal-content\"><div class=\"modal-header\"><button type=\"button\" class=\"close\" data-dismiss=\"modal\">&times;</button><h4 class=\"modal-title\">History</h4></div><div class=\"modal-body\"><div id=\"dataS\"></div><div id=\"Info\"></div></div><div class=\"modal-footer\"><button type=\"button\" class=\"btn btn-default\" data-dismiss=\"modal\">Close</button></div></div></div></div>");

                                          jQuery('#history').on('click', function(){

                                           $(".modal-body #dataS").html("");

                                          $('#myModal').modal('show');

                                                for (i = 0; i < data; i++) {
                                                   $(".modal-body #dataS").append("<button onclick=\"SAWs.history("+i+")\" id=\"buttonName" + i + "\" value=\"buttonValue\">" + i + "</button>");
                                                }


                                          })

                                      } else {

                                      }

                                   })
}

/*
  SAWs.displayDirectory - displays the passed directory tree sub to the DOM specifically

  This saves the displayed directory in the ugly global variable called globaldirectory.
  This is so that if SAWs.updateAndPostDirectory(null) is called then that directory is used. 
  We should instead somehow save all downloaded directories (without duplicates) so that all 
  all of them could be updated and posted.

  if sub has a bool variable named __Block in the root directory then a "Continue" button is presented 
  that when pressed updates __Block to false posts an update of the directory to the server and 
  then gets a refresh of data from the server.

  If any entry is changed or receives a keyup (like a carriage return in a text box) then the attribute selected is added to the object which then
  gets added to the JSON sent back to the server

*/
var globaldirectory = {}

SAWs.displayDirectory = function(sub,divEntry)
{

  console.log(sub);

  globaldirectory[divEntry] = sub
  if (sub.directories.SAWs_ROOT_DIRECTORY.variables.hasOwnProperty("__Block") && (sub.directories.SAWs_ROOT_DIRECTORY.variables.__Block.data[0] == "true")) {

    jQuery(divEntry).append("<center><input type=\"button\" value=\"Continue\" id=\"continue\"></center>")
    jQuery('#continue').on('click', function(){
                                      //$('#map-canvas').remove();
                                      SAWs.updateDirectoryFromDisplay(divEntry)
                                      sub.directories.SAWs_ROOT_DIRECTORY.variables.__Block.data = ["false"];
                                      SAWs.postDirectory(sub);
                                      jQuery(divEntry).html("");
                                      window.setTimeout(SAWs.getAndDisplayDirectory,1000,null,divEntry);
                                    })
  }





  SAWs.displayDirectoryRecursive(sub.directories,divEntry,0,"")
}

SAWs.tab = function(key,tab)
{
  for (i=0;i<tab;i++){
    jQuery("#"+key).append("&nbsp;&nbsp;&nbsp;&nbsp;")
  }
}

SAWs.history = function (field){

 jQuery.getJSON('/SAWs/historyGet/' + field,function(data){
                                                                    $("#Info").html("");
                                                                    SAWs.displayDirectoryAsHistory(data.directories,"Info",0,"Info");
                                                                })
}


SAWs.drawLine = function(long1,lat1,long2,lat2) {
    var lineone = [ new google.maps.LatLng(long1, lat1), new google.maps.LatLng(long2, lat2)];
    var line1 = new google.maps.Polyline({ path: lineone, geodesic: true, strokeColor: '#000000', strokeOpacity: 1.0, strokeWeight: 3 });
    line1.setMap(map);
}

SAWs.displayDirectoryRecursive = function(sub,divEntry,tab,fullkey)
{
  jQuery.each(sub,function(key,value){
                   fullkey = fullkey+key
                   if(jQuery("#"+fullkey).length == 0){
                     jQuery(divEntry).append("<div id =\""+fullkey+"\"></div>")
                     if (key != "SAWs_ROOT_DIRECTORY") {
			 SAWs.tab(fullkey,tab)
			 jQuery("#"+fullkey).append("<b>"+ key +"<b><br>")
                     }
                     jQuery.each(sub[key].variables, function(vKey, vValue) {
                                     if (vKey[0] != '_' || vKey[1] != '_' ) {
                                       SAWs.tab(fullkey,tab+1)
                                       if (vKey[0] != '_') {



                                            jQuery("#"+fullkey).append(vKey+":&nbsp;")




                                       }
				       for(j=0;j<sub[key].variables[vKey].data.length;j++){
				        if (sub[key].variables[vKey].data[j].toString() != "(null)"){	     
                                         if(sub[key].variables[vKey].alternatives.length == 0){

                                           if(sub[key].variables[vKey].dtype == "SAWs_MAP") {


                                             if (isMap) {
                                                                                          isMap = false;

                                                  $("body").append("<div id=\"map-canvas\" style=\"height: 50%;margin: 50px;padding: 0px\"></div>");

                                                                                                      var mapOptions = { zoom: 13, center: new google.maps.LatLng(41.696760, -87.948319), mapTypeId: google.maps.MapTypeId.ROADMAP };
                                                                                                      directionsDisplay = new google.maps.DirectionsRenderer();
                                                                                                      map = new google.maps.Map(document.getElementById('map-canvas'), mapOptions);
                                                                                                      directionsDisplay.setMap(map);
                                             }



                                             var res = sub[key].variables[vKey].data[0].split(":");

                                             var set1 = res[0].split(",");
                                             var set2 = res[1].split(",");






                                              SAWs.drawLine(set1[0],set1[1],set2[0],set2[1]);

                                                skipWrite = true;


                                           }

                                           if(sub[key].variables[vKey].dtype == "SAWs_BOOLEAN") {
                                             jQuery("#"+fullkey).append("<select id=\"data"+fullkey+vKey+j+"\">")
                                             jQuery("#data"+fullkey+vKey+j).append("<option value=\"true\">True</option> <option value=\"false\">False</option>")
                                           } else {

                                             if (false) {
                                                skipWrite = false;
                                             } else {

                                             //if(sub[key].variables[vKey].dtype =! "SAWs_MAP") {

                                             jQuery("#"+fullkey).append("<input type=\"text\" size=\""+(sub[key].variables[vKey].data[j].toString().length+1)+"\" id=\"data"+fullkey+vKey+j+"\" name=\"data\" \\>")

                                             jQuery("#data"+fullkey+vKey+j).keyup(function(obj) {
                                                                                   console.log( "Key up called "+key+vKey );
                                                                                   sub[key].variables[vKey].selected = 1;
                                                                                  });

                                             }


                                             //}
                                           }

                                           //if(sub[key].variables[vKey].dtype =! "SAWs_MAP") {


                                           jQuery("#data"+fullkey+vKey+j).val(sub[key].variables[vKey].data[j])
                                           jQuery("#data"+fullkey+vKey+j).change(function(obj) {
                                                                                                                              console.log( "Change called"+key+vKey );
                                                                                                                              sub[key].variables[vKey].selected = 1;
                                                                                                                            });


                                           //}


                                         } else {
                                           jQuery("#"+fullkey).append("<select id=\"data"+fullkey+vKey+j+"\">")
                                           jQuery("#data"+fullkey+vKey+j).append("<option value=\""+sub[key].variables[vKey].data[j]+"\">"+sub[key].variables[vKey].data[j]+"</option>")
                                           for(var l=0;l<sub[key].variables[vKey].alternatives.length;l++){
                                             jQuery("#data"+fullkey+vKey+j).append("<option value=\""+sub[key].variables[vKey].alternatives[l]+"\">"+sub[key].variables[vKey].alternatives[l]+"</option>")
                                           }
                                           jQuery("#"+fullkey).append("</select>")
                                           jQuery("#data"+fullkey+vKey+j).change(function(obj) {
                                                                                   console.log( "Change called"+key+vKey );
                                                                                   sub[key].variables[vKey].selected = 1;
                                                                                 });
                                         }
                                         if(sub[key].variables[vKey].mtype != "SAWs_WRITE") {
                                           jQuery("#data"+ fullkey+vKey+j).attr('readonly',true)
                                         } else {
                                           jQuery("#data"+ fullkey+vKey+j).attr('style',"color: #FF0000")
                                         }
					}    
                                       }
                                       jQuery("#"+fullkey).append("<br>")
                                     }                 
                                   })
                     if(typeof sub[key].directories != 'undefined'){
                       ++tab
                       SAWs.displayDirectoryRecursive(sub[key].directories,divEntry,tab,fullkey)
                       --tab
                     }
                   }
                })


                if (isMap) {

                }

}


/*
   Sends a POST of the directory to the server to update variable values that are writable
*/
SAWs.postDirectory = function(directory){
  var stringJSON = JSON.stringify(directory)
  /* could you jQuery.post()? */
  jQuery.ajax({type: 'POST',dataType: 'json',url: '/SAWs/*',data: {input: stringJSON}})
}
      
/*
   Sends the data from the html page to the C program. 
*/    
SAWs.updateAndPostDirectory = function(divEntry) {
  SAWs.updateDirectoryFromDisplay(divEntry)
  SAWs.postDirectory(globaldirectory[divEntry])
};

/*
   Updates a directory object from any changes in the HTML GUI
*/
SAWs.updateDirectoryFromDisplay = function(divEntry) {
   sub = globaldirectory[divEntry]
  SAWs.updateDirectoryFromDisplayRecursive(sub.directories,"")
}  


/*
   Updates a directory object from any changes in the HTML GUI
*/
SAWs.updateDirectoryFromDisplayRecursive = function(sub,fullkey) {
   jQuery.each(sub,function(key,value){
                     fullkey = fullkey+key
                     jQuery.each(sub[key].variables,function(vKey,vValue){
                                                       for(var k = 0; k<sub[key].variables[vKey].data.length; k++){
                                                         if(jQuery("#data"+fullkey+vKey +k).length != 0) {
                                                           var data1 = jQuery("#data"+ fullkey+vKey+k).val();
                    
                                                           /*Parse the data approriately*/
                                                           if(sub[key].variables[vKey].dtype == "SAWs_MAP"){

                                                           }

                                                           if(sub[key].variables[vKey].dtype == "SAWs_INT"){

                                                             if(!isNaN(parseInt(data1))){
                                                               sub[key].variables[vKey].data[k] = parseInt(data1)
                                                             }                   
                                                           } else if(sub[key].variables[vKey].dtype == "SAWs_DOUBLE" || sub[key].variables[vKey].dtype == "SAWs_FLOAT"){     
                                                             if(!isNaN(parseFloat(data1))){
                                                               sub[key].variables[vKey].data[k] = parseFloat(data1)
                                                             }
                                                           } else {

                                                             sub[key].variables[vKey].data[k] = data1
                                                           }
                                                         }
                                                       }
                                                     })
                     if(typeof sub[key].directories != 'undefined'){
                       SAWs.updateDirectoryFromDisplayRecursive(sub[key].directories,fullkey)
                     }
                   })
}

SAWs.displayDirectoryAsHistory = function(sub,divEntry,tab,fullkey) {


 jQuery.each(sub,function(key,value){
                   fullkey = fullkey+key
                   if(jQuery("#"+fullkey).length == 0){
                     jQuery("#Info").append("<div id =\""+fullkey+"\"></div>")
                     if (key != "SAWs_ROOT_DIRECTORY") {
			 SAWs.tab(fullkey,tab)
			 jQuery("#"+fullkey).append("<b>"+ key +"<b><br>")
                     }
                     jQuery.each(sub[key].variables, function(vKey, vValue) {
                                     if (vKey[0] != '_' || vKey[1] != '_' ) {
                                       SAWs.tab(fullkey,tab+1)
                                       if (vKey[0] != '_') {
                                         jQuery("#"+fullkey).append(vKey+":&nbsp;")
                                       }
				       for(j=0;j<sub[key].variables[vKey].data.length;j++){
				        if (sub[key].variables[vKey].data[j].toString() != "(null)"){
                                         if(sub[key].variables[vKey].alternatives.length == 0){
                                           if(sub[key].variables[vKey].dtype == "SAWs_BOOLEAN") {
                                             jQuery("#"+fullkey).append("<select id=\"data"+fullkey+vKey+j+"\">")
                                             jQuery("#data"+fullkey+vKey+j).append("<option value=\"true\">True</option> <option value=\"false\">False</option>")
                                           } else {
                                             jQuery("#"+fullkey).append("<input type=\"text\" size=\""+(sub[key].variables[vKey].data[j].toString().length+1)+"\" id=\"data"+fullkey+vKey+j+"\" name=\"data\" \\>")
                                             jQuery("#data"+fullkey+vKey+j).keyup(function(obj) {
                                                                                   console.log( "Key up called "+key+vKey );
                                                                                   sub[key].variables[vKey].selected = 1;
                                                                                  });
                                           }
                                           jQuery("#data"+fullkey+vKey+j).val(sub[key].variables[vKey].data[j])
                                           jQuery("#data"+fullkey+vKey+j).change(function(obj) {
                                                                                   console.log( "Change called"+key+vKey );
                                                                                   sub[key].variables[vKey].selected = 1;
                                                                                 });
                                         } else {
                                           jQuery("#"+fullkey).append("<select id=\"data"+fullkey+vKey+j+"\">")
                                           jQuery("#data"+fullkey+vKey+j).append("<option value=\""+sub[key].variables[vKey].data[j]+"\">"+sub[key].variables[vKey].data[j]+"</option>")
                                           for(var l=0;l<sub[key].variables[vKey].alternatives.length;l++){
                                             jQuery("#data"+fullkey+vKey+j).append("<option value=\""+sub[key].variables[vKey].alternatives[l]+"\">"+sub[key].variables[vKey].alternatives[l]+"</option>")
                                           }
                                           jQuery("#"+fullkey).append("</select>")
                                           jQuery("#data"+fullkey+vKey+j).change(function(obj) {
                                                                                   console.log( "Change called"+key+vKey );
                                                                                   sub[key].variables[vKey].selected = 1;
                                                                                 });
                                         }
                                         if(sub[key].variables[vKey].mtype != "SAWs_WRITE") {
                                           jQuery("#data"+ fullkey+vKey+j).attr('readonly',true)
                                         } else {
                                           jQuery("#data"+ fullkey+vKey+j).attr('style',"color: #FF0000")
                                         }
					}
                                       }
                                       jQuery("#"+fullkey).append("<br>")
                                     }
                                   })
                     if(typeof sub[key].directories != 'undefined'){
                       ++tab
                       SAWs.displayDirectoryAsHistory(sub[key].directories,divEntry,tab,fullkey)
                       --tab
                     }
                   }
                })



}