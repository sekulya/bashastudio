const express = require('express');
const app = express();
const PORT = 3000;

// Allow CORS (for frontend communication)
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  next();
});

// Example API endpoint
app.get('/api/data', (req, res) => {
  res.json({ message: "Hello from Lightsail!" });
});

// ▼▼▼ ADD YOUR LOGIN ROUTE HERE ▼▼▼
app.post('/login', (req, res) => { 
  res.send('Login endpoint working!');
});
// ▲▲▲ ABOVE OTHER ROUTES ▲▲▲

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Also accessible via http://43.202.24.48:${PORT}`);
  console.log(`Server running on bashastudios.online:${PORT}`);
});