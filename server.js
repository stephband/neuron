import { serveDir } from "jsr:@std/http/file-server";

console.log("Starting WebGPU Neuron Server at http://localhost:8000 ...");
Deno.serve({ port: 8000 }, (req) => {
  return serveDir(req, { fsRoot: "." });
});
