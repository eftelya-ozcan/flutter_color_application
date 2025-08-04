import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:io';
import 'package:image_picker/image_picker.dart';
import 'dart:convert';
import 'package:flutter_tts/flutter_tts.dart';

void main() => runApp(ColorDetectorApp());

class ColorDetectorApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(home: ColorHomePage());
  }
}

class ColorHomePage extends StatefulWidget {
  @override
  _ColorHomePageState createState() => _ColorHomePageState();
}

class _ColorHomePageState extends State<ColorHomePage> {
  File? _image;
  String? _predictedColor;

  FlutterTts flutterTts = FlutterTts();

  Future<void> _pickImageAndSend() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.camera);

    if (pickedFile == null) return;

    final imageFile = File(pickedFile.path);
    setState(() => _image = imageFile);

    final request = http.MultipartRequest(
      'POST',
      Uri.parse('http://192.168.1.144/predict-color'),
    );
    request.files.add(
      await http.MultipartFile.fromPath('image', imageFile.path),
    );
    final response = await request.send();

    if (response.statusCode == 200) {
      final responseData = await http.Response.fromStream(response);
      final json = jsonDecode(responseData.body);
      setState(() {
        _predictedColor = json['predicted_color'];
      });
      await flutterTts.setLanguage("tr-TR");
      await flutterTts.speak("Renk: ${_predictedColor!}");
    } else {
      setState(() => _predictedColor = 'Hata oluştu');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Renk Tanıyıcı')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (_image != null) Image.file(_image!, height: 200),
            ElevatedButton(
              onPressed: _pickImageAndSend,
              child: Text("Fotoğraf Çek ve Renk Tahmin Et"),
            ),
            if (_predictedColor != null)
              Text("Tahmin Edilen Renk: $_predictedColor"),
          ],
        ),
      ),
    );
  }
}
