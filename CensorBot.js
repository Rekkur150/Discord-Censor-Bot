var Discord = require("discord.js")
var PathToStringChecker = "./CheckString.py"

var HateSpeechThreshHold = 0.9;


//Discord
const token = '';
var CensorBot = new Discord.Client();


function convert_to_float(a) {
	var floatValue = +(a);

	return floatValue;
}


CensorBot.on('message', message => {

	const CreateProcess = require("child_process").spawn;
	const StringChecker = CreateProcess('python3',[PathToStringChecker, message.content])
	
	StringChecker.stdout.on('data', data => {
		var LikelyHoodToBeHateSpeech = convert_to_float(data.toString().replace(/\[|\]/g, ''));
		if (LikelyHoodToBeHateSpeech >= HateSpeechThreshHold) {
			console.log("HATE SPEECH DETECTED");
		} else {
			console.log("HATE SPEECH NOT DETECTED");
		}

	});
});
