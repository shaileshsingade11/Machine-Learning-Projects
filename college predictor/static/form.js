import express from "express";
import bodyParser from "body-parser";
import { dirname } from "path";
import { fileURLToPath } from "url";
const __dirname = dirname(fileURLToPath(import.meta.url));

const app = express();
const port = 3000;


// adding a middleware
app.use(bodyParser.urlencoded({extended:true}));


app.get("/", (req, res) => {
  console.log(__dirname + "/template/index.html");
  res.sendFile(__dirname + "/template/index.html");
});

app.post("/submit" , (req,res)=>{
  console.log(req.body);
})

app.listen(port, () => {
  console.log(`Listening on port ${port}`);
});
