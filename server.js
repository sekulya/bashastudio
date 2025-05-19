const express = require('express');
const app = express();
const PORT = 3000;

// CORS middleware
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  next();
});

// Root route
app.get('/', (req, res) => {
  res.send('Server is running!');
});

// API endpoint
app.get('/api/data', (req, res) => {
  res.json({ message: "Hello from Lightsall!" });
});

// Login route
app.post('/login', (req, res) => {
  res.send('Login endpoint working!');
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Also accessible via http://43.201.59.108:${PORT}`);
  console.log(`Server running on bashastudios.online:${PORT}`);
});
