using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;

namespace WebApplication1.Pages;

public class IndexModel : PageModel
{
    private readonly ILogger<IndexModel> _logger;

    public IndexModel(ILogger<IndexModel> logger)
    {
        _logger = logger;
    }

    [BindProperty]
    public IFormFile? UploadedImage { get; set; }

    public string? ImageUrl { get; set; }

    public void OnGet()
    {

    }

    public async Task<IActionResult> OnPostAsync()
    {
        if (UploadedImage != null && UploadedImage.Length > 0)
        {
            // Verificar se é uma imagem
            var allowedExtensions = new[] { ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".svg", ".ico", ".avif", ".heic", ".heif", ".jfif", ".jpe", ".jfi" };
            var fileExtension = Path.GetExtension(UploadedImage.FileName).ToLowerInvariant();

            if (!allowedExtensions.Contains(fileExtension))
            {
                ModelState.AddModelError("UploadedImage", "Por favor, selecione um arquivo de imagem válido (.jpg, .jpeg, .png, .gif, .bmp, .webp, .tiff, .svg, .ico, .avif, .heic, .heif, .jfif)");
                return Page();
            }

            // Criar o diretório uploads se não existir
            var uploadsFolder = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot", "uploads");
            if (!Directory.Exists(uploadsFolder))
            {
                Directory.CreateDirectory(uploadsFolder);
            }

            // Gerar nome único para o arquivo
            var fileName = Guid.NewGuid().ToString() + fileExtension;
            var filePath = Path.Combine(uploadsFolder, fileName);

            // Salvar o arquivo
            using (var stream = new FileStream(filePath, FileMode.Create))
            {
                await UploadedImage.CopyToAsync(stream);
            }

            // Definir a URL da imagem para exibição
            ImageUrl = $"/uploads/{fileName}";
        }

        return Page();
    }
}
